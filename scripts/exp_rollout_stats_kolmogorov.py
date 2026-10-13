"""Rollout statistics of crop-trained Kolmogorov models vs truth (2026-10-04, P9 diagnosis).

P9 (2D patches of a 6x6-box domain) gives D_KY 25.5 / 53.9 / 107.3 at 1/2/3x (truth 23.3 / ~50 /
75.2): superlinear growth. Question: does the model's free rollout at large Lx drift into a state
statistically different from the truth (more energetic, different large-scale content), which would
explain the extra chaos? Compares, on 2pi-tall boxes at 1/2/3x:
  - decoded vorticity std (physical units), time-resolved (early vs late windows)
  - energy share in streamwise modes with wavelength > 2pi (beyond the native box)
  - energy share in modes |k| < 2 (large scales) and at the forcing scale |k| in [3,5]
  - mean |z| per latent channel (latent drift)
for a model checkpoint pair and for the true PDE from the same initial condition.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from ks_latent.config import KolmogorovConfig
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.models.kolmogorov_fields import encode_input, field0
from ks_latent.solver.kolmogorov import integrate_kolmogorov, spinup_kolmogorov

OUT = Path("artifacts/experiments/rollout_stats")


def field_stats(w: np.ndarray, Lx: float, Ly: float) -> dict:
    """w: (T, NX, NY) physical vorticity."""
    T, NX, NY = w.shape
    kx = np.fft.fftfreq(NX, d=Lx / NX) * 2 * np.pi
    ky = np.fft.fftfreq(NY, d=Ly / NY) * 2 * np.pi
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2
    K = np.sqrt(K2)
    K2s = np.where(K2 > 0, K2, 1.0)
    P = (np.abs(np.fft.fft2(w)) ** 2 / K2s)  # kinetic-energy density per mode
    P[:, 0, 0] = 0
    tot = P.sum(axis=(1, 2))
    m = np.minimum(np.arange(NX), NX - np.arange(NX))
    S = int(round(Lx / (2 * np.pi)))
    h = T // 2
    beyond = P[:, (m >= 1) & (m < S), :].sum(axis=(1, 2)) / tot if S > 1 else np.zeros(T)
    large = (P * (K < 2)[None]).sum(axis=(1, 2)) / tot
    forc = (P * ((K >= 3) & (K <= 5))[None]).sum(axis=(1, 2)) / tot
    ts = large[::max(1, T // 100)]
    # R12: correlation between the field and itself shifted by one native box (2pi) in x; ~1 if the
    # rollout has locked into a 2pi-periodic state (the larger domain behaves like copies of the box)
    wl = w[h:] - w[h:].mean(axis=(1, 2), keepdims=True)
    sh = np.roll(wl, NX // S, axis=1) if S > 1 else wl
    box_corr = float((wl * sh).sum() / np.sqrt((wl ** 2).sum() * (sh ** 2).sum()))
    return {
        "box_period_corr_late": box_corr,
        "ts_share_k_lt_2": [round(float(v), 4) for v in ts],
        "std_early": float(w[:h].std()), "std_late": float(w[h:].std()),
        "kinetic_energy_early": float(tot[:h].mean() / (NX * NY) ** 2),
        "kinetic_energy_late": float(tot[h:].mean() / (NX * NY) ** 2),
        "share_beyond_box_late": float(beyond[h:].mean()),
        "share_k_lt_2_late": float(large[h:].mean()),
        "share_forcing_k3_5_late": float(forc[h:].mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="checkpoint tag, e.g. p9_a005 or p2_a005")
    p.add_argument("--scales", default="1,2,3")
    p.add_argument("--n-steps", type=int, default=1000, help="model steps (dt_snap each)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-truth", action="store_true")
    p.add_argument("--y-scale", type=int, default=1, help="Ly = y_scale * 2pi (model-only geometry tests)")
    a = p.parse_args()
    A = Path("artifacts")
    ae_ckpt, prop_ckpt = A / f"stage1_ae_kolmogorov_crop_{a.tag}.pt", A / f"stage2_prop_kolmogorov_crop_{a.tag}.pt"
    ck = torch.load(ae_ckpt, map_location="cpu", weights_only=False)
    mean, std = ck["normalization_mean"], ck["normalization_std"]
    with h5py.File(ck["train_dataset"], "r") as f:
        md = dict(f["metadata"].attrs)
    res = {"tag": a.tag, "train_dataset": ck["train_dataset"], "norm_std": std, "seed": a.seed,
           "n_steps": a.n_steps, "dt_snap": float(md["dt_snap"]), "scales": {}}
    for S in [int(s) for s in a.scales.split(",")]:
        t0 = time.time()
        Sy = a.y_scale
        NX, NY, nsx, nsy = 128 * S, 128 * Sy, 8 * S, 8 * Sy
        cfg = KolmogorovConfig(Lx=2 * math.pi * S, Ly=2 * math.pi * Sy, NX=NX, NY=NY, nu=float(md["nu"]),
                               alpha=float(md["alpha"]), forcing_n=int(md["forcing_n"]), dt=float(md["dt"]),
                               spinup_time=100.0, seed=a.seed)
        u0 = spinup_kolmogorov(cfg, np.random.default_rng(a.seed))
        ae = load_autoencoder_checkpoint_resized_2d(str(ae_ckpt), new_n_sites_x=nsx, new_n_sites_y=nsy,
                                                    new_NX=NX, new_NY=NY)[0].eval()
        from ks_latent.models.memory_prop import is_memory_checkpoint, load_memory_prop_resized
        memory = is_memory_checkpoint(prop_ckpt)
        c = ae.cfg.local_channels
        if memory:  # Z10a: warm up the memory on W truth snapshots (z forced), roll out from the last one
            prop, _, pck = load_memory_prop_resized(str(prop_ckpt), nsx, nsy)
            prop.eval()
            Wm = int(pck.get("memory_warmup", 8))
            ev = int(round(float(md["dt_snap"]) / cfg.dt))
            hist = integrate_kolmogorov(u0, cfg, n_steps=Wm * ev, snapshot_every=ev)  # (Wm+1, NX, NY)
            u0 = hist[-1]  # the truth comparison starts from the same state as the free rollout
            with torch.no_grad():
                zh = torch.cat([ae.encode(encode_input(ae, (hh - mean) / std, ck, cfg.Lx, cfg.Ly)) for hh in hist])
                x = prop.warmup(zh.unsqueeze(0))
                roll = prop.encoder_part(prop.rollout(x, x, k=a.n_steps)[0])
                dec = torch.cat([ae.decode(roll[i:i + 100]) for i in range(0, a.n_steps, 100)])
        else:
            prop = load_propagator_checkpoint_resized_2d(str(prop_ckpt), new_n_sites_x=nsx, new_n_sites_y=nsy)[0].eval()
            with torch.no_grad():
                z = ae.encode(encode_input(ae, (u0 - mean) / std, ck, cfg.Lx, cfg.Ly))  # R15: wuv-aware
                roll = prop.rollout(z, z, k=a.n_steps)[0]  # (k, d)
                dec = torch.cat([ae.decode(roll[i:i + 100]) for i in range(0, a.n_steps, 100)])
        w_model = field0(dec, NX, NY, getattr(ae.cfg, "n_fields", 1)).numpy() * std + mean  # R15: omega channel
        zs = roll.numpy().reshape(a.n_steps, nsx, nsy, c)
        h = a.n_steps // 2
        entry = {"model": field_stats(w_model, cfg.Lx, cfg.Ly),
                 "latent_abs_mean_by_channel_early": np.abs(zs[:h]).mean(axis=(0, 1, 2)).tolist(),
                 "latent_abs_mean_by_channel_late": np.abs(zs[h:]).mean(axis=(0, 1, 2)).tolist(),
                 "latent_max_abs": float(np.abs(zs).max())}
        if a.with_truth:
            n_solver = int(round(a.n_steps * float(md["dt_snap"]) / cfg.dt))
            every = int(round(float(md["dt_snap"]) / cfg.dt))
            w_true = integrate_kolmogorov(u0, cfg, n_steps=n_solver, snapshot_every=every)[1:]
            entry["truth"] = field_stats(w_true, cfg.Lx, cfg.Ly)
        entry["wall_s"] = time.time() - t0
        res["scales"][f"{S}x{a.y_scale}"] = entry
        print(f"[rollout] {a.tag} x{S}: " + json.dumps({k: round(v, 4) for k, v in entry["model"].items() if isinstance(v, float)})
              + (" | truth: " + json.dumps({k: round(v, 4) for k, v in entry["truth"].items() if isinstance(v, float)}) if a.with_truth else ""),
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}_seed{a.seed}_n{a.n_steps}_y{a.y_scale}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
