"""Task A6 (docs/full_paper_tasks.md): decoded energy spectra and reconstruction quality.

For one system and one domain size:
  * truth: a long PDE trajectory from a spun-up IC (IC seed 0), snapshots at the model's dt_snap;
  * model: free rollout of the transferred model from the encoded (standardized) first snapshot,
    decoded to physical units; statistics over steps [discard, n_steps);
  * reconstruction: D(E(u)) of every truth snapshot (no propagation).
Reports time-averaged energy spectra (1D: |u_hat|^2; 2D KS: shell-averaged |u_hat|^2;
Kolmogorov: shell-averaged kinetic energy |w_hat|^2/|k|^2), per-band energy ratios
model/truth and reconstruction/truth, reconstruction MSE (standardized units) and its
fraction of the field variance, and decoded field std / max|u|.

Usage:
  python scripts/eval_spectra_recon.py --system ks1d --scale 1      # L = 100 * scale
  python scripts/eval_spectra_recon.py --system ks2d --scale 1.5    # L = 44 * scale per side
  python scripts/eval_spectra_recon.py --system kolm --scale 3      # Lx = 2pi * scale, Ly = 2pi
Writes artifacts/experiments/spectra/<system>_x<scale>.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from ks_latent.config import KolmogorovConfig, KSConfig2D
from ks_latent.models import (
    load_autoencoder_checkpoint_resized,
    load_autoencoder_checkpoint_resized_2d,
    load_propagator_checkpoint_resized,
    load_propagator_checkpoint_resized_2d,
)
from ks_latent.solver.kolmogorov import integrate_kolmogorov, spinup_kolmogorov
from ks_latent.solver.ks import KSConfig, integrate, spinup
from ks_latent.solver.ks2d import integrate2d, spinup2d

OUT = Path("artifacts/experiments/spectra")
CKPT = {
    "ks1d": ("artifacts/stage1_ae_patched_full_section224_ks_localfield_d48_localmlp.pt",
             "artifacts/stage2_prop_patched_full_section224_ks_localfield_d48_localmlp_warmstart_k12_300ep.pt"),
    "ks2d": ("artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt", "artifacts/stage2_prop_kolmogorov_crop_p7b_ks2d.pt"),
}
# Kolmogorov alpha=0.1: the main-text model chosen by the truth-free rule of task A10. The
# first A6 run (before A10) used k_xs01_v5; those files are kept as kolm_x<s>_k_xs01_v5.json.
_A01 = json.loads(Path("artifacts/experiments/a10_selection/decision.json").read_text())["selected"]
CKPT["kolm"] = (f"artifacts/stage1_ae_kolmogorov_crop_{_A01}.pt", f"artifacts/stage2_prop_kolmogorov_crop_{_A01}.pt")
# Band edges in |k| (physical wavenumber). KS: around the most unstable k = 1/sqrt(2).
# Kolmogorov: octaves around the forcing wavenumber n = 4.
BANDS = {"ks1d": [0, 0.35, 0.7, 1.05, 1.4, np.inf], "ks2d": [0, 0.35, 0.7, 1.05, 1.4, np.inf],
         "kolm": [0, 1, 2, 4, 8, 16, np.inf]}


def spectrum_1d(u: np.ndarray, L: float):
    """u: (T, NX). Returns k (>=0) and time-averaged |u_hat|^2/NX^2."""
    nx = u.shape[1]
    k = 2 * np.pi * np.fft.rfftfreq(nx, d=L / nx)
    e = np.mean(np.abs(np.fft.rfft(u, axis=1)) ** 2, axis=0) / nx**2
    return k, e


def spectrum_2d(u: np.ndarray, Lx: float, Ly: float, kinetic: bool):
    """u: (T, NX, NY). Shell-averaged (sum over shells of width 2pi/max(Lx,Ly)) spectrum."""
    T, nx, ny = u.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * np.fft.rfftfreq(ny, d=Ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)
    w = np.full(K.shape, 2.0); w[:, 0] = 1.0
    if ny % 2 == 0:
        w[:, -1] = 1.0  # rfft half-plane weights (Nyquist column counted once)
    p = np.zeros(K.shape)
    for t in range(T):  # streaming time average keeps memory flat
        p += np.abs(np.fft.rfft2(u[t])) ** 2
    p = p * w / (T * (nx * ny) ** 2)
    if kinetic:
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(K > 0, p / K**2, 0.0)
    dk = 2 * np.pi / max(Lx, Ly)
    idx = np.rint(K / dk).astype(int)
    e = np.bincount(idx.ravel(), weights=p.ravel())
    return np.arange(e.size) * dk, e


def band_ratios(k, e_ref, e_other, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (k >= lo) & (k < hi) & (k > 0)
        ref = float(e_ref[m].sum())
        out.append({"k_lo": float(lo), "k_hi": float(hi) if np.isfinite(hi) else None,
                    "energy_ref": ref, "energy_other": float(e_other[m].sum()),
                    "ratio": float(e_other[m].sum() / ref) if ref > 0 else None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(CKPT), required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--discard", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fields-out", default=None,
                    help="write truth/model fields to this .npz (for figures) instead of the JSON")
    ap.add_argument("--field-stride", type=int, default=1,
                    help="keep every n-th snapshot in --fields-out (2D: also used to thin the record)")
    a = ap.parse_args()
    t0 = time.time()
    ae_path, prop_path = CKPT[a.system]

    if a.system == "ks1d":
        with h5py.File("artifacts/datasets/stage1_trajectories_dtsnap1.h5", "r") as f:
            mean = float(f["metadata"].attrs["normalization_mean"]); std = float(f["metadata"].attrs["normalization_std"])
        L = 100.0 * a.scale; nx = int(round(256 * a.scale)); ns = int(round(16 * a.scale))
        ae = load_autoencoder_checkpoint_resized(ae_path, new_n_sites=ns, new_NX=nx)[0].eval()
        d = ns * ae.cfg.local_channels
        prop = load_propagator_checkpoint_resized(prop_path, new_d_latent=d, new_n_tokens=ns)[0].eval()
        cfg = KSConfig(L=L, NX=nx, dt=0.05, snapshot_every=20, spinup_time=500.0, seed=a.seed)
        truth = integrate(spinup(cfg, np.random.default_rng(a.seed)), cfg, n_steps=a.n_steps * 20)[: a.n_steps]
        shape, dt_snap, geom = (nx,), 1.0, {"L": L, "NX": nx, "n_sites": ns}
    else:
        ck = torch.load(ae_path, map_location="cpu", weights_only=False)
        mean, std = float(ck["normalization_mean"]), float(ck["normalization_std"])
        if a.system == "ks2d":
            L = 44.0 * a.scale; nx = ny = int(round(128 * a.scale)); nsx = nsy = int(round(8 * a.scale))
            cfg = KSConfig2D(Lx=L, Ly=L, NX=nx, NY=ny, spinup_time=200.0, seed=a.seed)
            cache = Path(f"artifacts/cache/ks2d_spinup_L{L:g}_NX{nx}_dt0.02_T200_seed{a.seed}.npy")
            u0 = np.load(cache) if cache.exists() else spinup2d(cfg, np.random.default_rng(a.seed))
            dt_snap = cfg.dt * cfg.snapshot_every
            every = cfg.snapshot_every
            truth = integrate2d(u0, cfg, n_steps=a.n_steps * every)[: a.n_steps]
            geom = {"Lx": L, "Ly": L, "NX": nx, "NY": ny, "spinup_cache_used": cache.exists()}
        else:
            with h5py.File(ck["train_dataset"], "r") as f:
                md = dict(f["metadata"].attrs)
            S = int(round(a.scale)); nx, ny, nsx, nsy = 128 * S, 128, 8 * S, 8
            cfg = KolmogorovConfig(Lx=2 * math.pi * S, Ly=2 * math.pi, NX=nx, NY=ny, nu=float(md["nu"]),
                                   alpha=float(md["alpha"]), forcing_n=int(md["forcing_n"]), dt=float(md["dt"]),
                                   spinup_time=100.0, seed=a.seed)
            u0 = spinup_kolmogorov(cfg, np.random.default_rng(a.seed))
            dt_snap = float(md["dt_snap"]); every = int(round(dt_snap / cfg.dt))
            truth = integrate_kolmogorov(u0, cfg, n_steps=a.n_steps * every, snapshot_every=every)[: a.n_steps]
            geom = {"Lx": cfg.Lx, "Ly": cfg.Ly, "NX": nx, "NY": ny, "alpha": float(md["alpha"])}
        ae = load_autoencoder_checkpoint_resized_2d(ae_path, new_n_sites_x=nsx, new_n_sites_y=nsy,
                                                    new_NX=nx, new_NY=ny)[0].eval()
        prop = load_propagator_checkpoint_resized_2d(prop_path, new_n_sites_x=nsx, new_n_sites_y=nsy)[0].eval()
        shape = (nx, ny)

    flat = lambda x: torch.tensor(((x - mean) / std).reshape(x.shape[0], -1), dtype=torch.float32)  # noqa: E731
    with torch.no_grad():
        z0 = ae.encode(flat(truth[:1]))
        roll = torch.cat([z0, prop.rollout(z0, z0, k=a.n_steps - 1)[0]], 0)
        model = torch.cat([ae.decode(roll[i:i + 100]) for i in range(0, a.n_steps, 100)]).numpy()
        recon = torch.cat([ae.decode(ae.encode(flat(truth[i:i + 100]))) for i in range(0, a.n_steps, 100)]).numpy()
    model = model.reshape(-1, *shape).astype(np.float64) * std + mean
    recon = recon.reshape(-1, *shape).astype(np.float64) * std + mean
    finite = bool(np.isfinite(model).all())

    if a.fields_out:
        sl = slice(0, None, a.field_stride)
        np.savez_compressed(a.fields_out, truth=truth[sl].astype(np.float32), model=model[sl].astype(np.float32),
                            recon=recon[sl].astype(np.float32), dt_snap=dt_snap * a.field_stride,
                            geom=json.dumps(geom), ae=ae_path, prop=prop_path, seed=a.seed)
        print(f"wrote {a.fields_out} ({time.time() - t0:.0f}s)")
        return

    win = slice(a.discard, None)
    if a.system == "ks1d":
        k, e_t = spectrum_1d(truth[win], L)
        _, e_m = spectrum_1d(model[win], L) if finite else (k, np.full_like(e_t, np.nan))
        _, e_r = spectrum_1d(recon, L)
        _, e_t_all = spectrum_1d(truth, L)
    else:
        kin = a.system == "kolm"
        k, e_t = spectrum_2d(truth[win], geom["Lx"], geom["Ly"], kin)
        _, e_m = spectrum_2d(model[win], geom["Lx"], geom["Ly"], kin) if finite else (k, np.full_like(e_t, np.nan))
        _, e_r = spectrum_2d(recon, geom["Lx"], geom["Ly"], kin)
        _, e_t_all = spectrum_2d(truth, geom["Lx"], geom["Ly"], kin)

    err_std = (recon - truth) / std
    rec = {
        "system": a.system, "scale": a.scale, "seed": a.seed, **geom, "dt_snap": dt_snap,
        "n_steps": a.n_steps, "discard": a.discard, "ae_checkpoint": ae_path, "prop_checkpoint": prop_path,
        "normalization_mean": mean, "normalization_std": std, "rollout_finite": finite,
        "recon_mse_standardized": float(np.mean(err_std**2)),
        "recon_mse_over_field_variance": float(np.mean((recon - truth) ** 2) / truth.var()),
        "field_std_truth": float(truth[win].std()), "field_std_model": float(model[win].std()) if finite else None,
        "max_abs_truth": float(np.abs(truth).max()), "max_abs_model": float(np.abs(model).max()) if finite else None,
        "k": k.tolist(), "spectrum_truth": e_t.tolist(), "spectrum_model": e_m.tolist(),
        "spectrum_recon": e_r.tolist(), "spectrum_truth_all_snapshots": e_t_all.tolist(),
        "band_ratio_model_over_truth": band_ratios(k, e_t, e_m, BANDS[a.system]),
        "band_ratio_recon_over_truth": band_ratios(k, e_t_all, e_r, BANDS[a.system]),
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "wall_s": time.time() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.system}_x{a.scale:g}.json"
    out.write_text(json.dumps(rec, indent=2))
    br = " ".join(f"{b['ratio']:.2f}" for b in rec["band_ratio_model_over_truth"] if b["ratio"] is not None)
    rr = " ".join(f"{b['ratio']:.2f}" for b in rec["band_ratio_recon_over_truth"] if b["ratio"] is not None)
    print(f"[{out.name}] recon MSE {rec['recon_mse_standardized']:.4g} ({100*rec['recon_mse_over_field_variance']:.2f}% of var) "
          f"| std truth/model {rec['field_std_truth']:.3f}/{rec['field_std_model']} | band model/truth: {br} "
          f"| band recon/truth: {rr} ({rec['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
