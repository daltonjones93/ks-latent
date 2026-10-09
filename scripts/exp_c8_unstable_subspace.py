"""C8 (docs/ks_local_field_ltransfer.tex Sec. c8c12): where do the missing
dimensions live? alpha=0.05, Lx = 3 x 2pi.

  --part model     P2's crop-trained model at 3x: Benettin from a normalized
                   IC (seed 0, exp_crop_eval settings), saving the final
                   latent tangent basis; its leading vectors are mapped to
                   physical space through the decoder Jacobian at the final
                   state.
  --part autocorr  Energy in the lowest streamwise modes (m = 1,2,3 at 3x;
                   m=3 is the 2pi box's fundamental) along a true trajectory
                   and along a decoded model rollout: mean level and
                   e-folding time (mechanism (b)).
  --part analyze   Needs the truth Benettin checkpoint
                   (artifacts/experiments/crop/truth_a005_x3_seed0.ckpt.npz,
                   written by exp_crop_pipeline's truth task). Orthonormalizes
                   the leading n_positive vectors of each unstable subspace in
                   physical space and measures the share of their variance in
                   streamwise wavenumber bands, then applies the
                   pre-registered rule.

Bands at 3x (Lx = 6pi, streamwise index m, wavelength 6pi/m):
  beyond_box: m in {1, 2}   (lambda > 2pi; these modes do not exist at native size)
  non_box:    m % 3 != 0    (any mode not representable in the 2pi box)
  local:      everything else
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp

from ks_latent.analysis.lyapunov import _make_single_state_step_fn, benettin
from ks_latent.config import KolmogorovConfig
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.kolmogorov import integrate_kolmogorov, spinup_kolmogorov

REPO = Path(__file__).resolve().parents[1]
A = REPO / "artifacts"
OUT = A / "experiments" / "c8"
AE = A / "stage1_ae_kolmogorov_crop_p2_a005.pt"
PROP = A / "stage2_prop_kolmogorov_crop_p2_a005.pt"
TRUTH_CKPT = A / "experiments/crop/truth_a005_x3_seed0.ckpt.npz"
TRUTH_JSON = A / "experiments/crop/truth_a005_x3_seed0.json"
S = 3
NX, NY = 128 * S, 128


def load_models():
    ck = torch.load(AE, map_location="cpu", weights_only=False)
    ae = load_autoencoder_checkpoint_resized_2d(str(AE), new_n_sites_x=8 * S, new_n_sites_y=8, new_NX=NX, new_NY=NY)[0]
    prop = load_propagator_checkpoint_resized_2d(str(PROP), new_n_sites_x=8 * S, new_n_sites_y=8)[0]
    return ae.eval(), prop.eval(), ck["normalization_mean"], ck["normalization_std"]


def kcfg(seed=0):
    return KolmogorovConfig(Lx=2 * math.pi * S, Ly=2 * math.pi, NX=NX, NY=NY, nu=0.01, alpha=0.05, forcing_n=4,
                            dt=0.01, spinup_time=100.0, seed=seed)


def part_model():
    ae, prop, mean, std = load_models()
    u0 = (spinup_kolmogorov(kcfg(), np.random.default_rng(0)) - mean) / std
    with torch.no_grad():
        z0 = ae.encode(torch.tensor(u0.reshape(1, -1), dtype=torch.float32))[0].numpy().astype(np.float64)
    d = prop.cfg.d_latent
    ckpt = OUT / "model_p2_x3_seed0.ckpt.npz"
    t0 = time.time()
    res = benettin(_make_single_state_step_fn(prop), z0, n_directions=d, n_steps=2000, qr_every=10,
                   dt_per_step=0.2, warmup_steps=200, rng=np.random.default_rng(0), max_abs_state=1e3,
                   checkpoint_path=str(ckpt), checkpoint_every=50)
    with np.load(ckpt) as c:
        zT, Q = c["state"], c["tangent"]  # Q rows: orthonormal latent tangent basis, Benettin order
    n_keep = min(d, res.n_positive + 10)
    zt = torch.tensor(zT, dtype=torch.float32)
    phys = []
    for v in Q[:n_keep]:
        _, jv = jvp(lambda z: ae.decode(z.unsqueeze(0))[0], (zt,), (torch.tensor(v, dtype=torch.float32),))
        phys.append(jv.detach().numpy())
    np.savez(OUT / "model_p2_x3_seed0_phys_vectors.npz", vectors=np.stack(phys), exponents=res.exponents,
             n_positive=res.n_positive, D_KY=res.kaplan_yorke_dimension)
    print(f"[c8 model] D_KY={res.kaplan_yorke_dimension:.2f} n_pos={res.n_positive} "
          f"(eval run gave 72.0) [{time.time() - t0:.0f}s]", flush=True)


def band_energy_series(fields):
    """fields: (T, NX, NY) -> dict m -> (T,) energy in streamwise index m (both +-m)."""
    F = np.fft.fft(fields, axis=1)
    return {m: (np.abs(F[:, m]) ** 2 + np.abs(F[:, -m]) ** 2).sum(axis=-1) for m in (1, 2, 3)}


def efold(x):
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    below = np.where(ac < 1 / math.e)[0]
    return int(below[0]) if len(below) else len(x)


def part_autocorr(n_snap=500):
    ae, prop, mean, std = load_models()
    cfg = kcfg(seed=1)
    u0 = spinup_kolmogorov(cfg, np.random.default_rng(1))
    truth = (integrate_kolmogorov(u0, cfg, n_steps=n_snap * cfg.snapshot_every)[1:] - mean) / std
    with torch.no_grad():
        z = ae.encode(torch.tensor(truth[:1].reshape(1, -1), dtype=torch.float32))
        roll = prop.rollout(z, z, k=n_snap)[0]
        model = np.concatenate([ae.decode(roll[i:i + 50]).numpy() for i in range(0, n_snap, 50)]).reshape(-1, NX, NY)
    out = {}
    for name, f in (("truth", truth), ("model", model[100:])):
        bands = band_energy_series(f)
        out[name] = {f"m{m}": {"mean_energy": float(e.mean()), "efold_snapshots": efold(e),
                               "efold_time": efold(e) * cfg.dt_snap} for m, e in bands.items()}
    (OUT / "autocorr_x3.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


def band_fractions(vectors, n):
    """Orthonormalize the first n vectors (nested-span preserving QR) and return
    the share of total variance per streamwise band."""
    q, _ = np.linalg.qr(vectors[:n].reshape(n, -1).T)
    V = q.T.reshape(n, NX, NY)
    P = (np.abs(np.fft.fft(V, axis=1)) ** 2).sum(axis=(0, 2))  # (NX,)
    m = np.minimum(np.arange(NX), NX - np.arange(NX))
    tot = P.sum()
    return {"beyond_box": float(P[(m >= 1) & (m <= S - 1)].sum() / tot),
            "non_box": float(P[(m % S) != 0].sum() / tot),
            "m0": float(P[m == 0].sum() / tot)}


def part_analyze():
    truth = json.loads(TRUTH_JSON.read_text())
    with np.load(TRUTH_CKPT) as c:
        Qt = c["tangent"]
    n_t = int(truth["n_positive"])
    mod = np.load(OUT / "model_p2_x3_seed0_phys_vectors.npz")
    n_m = int(mod["n_positive"])
    ft = band_fractions(Qt, n_t)
    fm = band_fractions(mod["vectors"], n_m)
    missing = n_t - n_m
    res = {"truth": {"n_positive": n_t, "D_KY": truth["D_KY"], "fractions": ft},
           "model": {"n_positive": n_m, "D_KY": float(mod["D_KY"]), "fractions": fm},
           "missing_dims": missing}
    for band in ("beyond_box", "non_box"):
        excess = ft[band] * n_t - fm[band] * n_m
        res[f"excess_dims_{band}"] = excess
        res[f"excess_share_{band}"] = excess / missing if missing > 0 else None
    share = res["excess_share_beyond_box"]
    res["verdict"] = ("no deficit at 3x (model n_positive >= truth)" if missing <= 0 else
                      "LARGE-SCALE (mechanisms a/c confirmed; C9/C10 first)" if share >= 0.5 else
                      "LOCAL (deficit mostly at small scales; C11 statistics first)")
    (OUT / "c8_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["model", "autocorr", "analyze"], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"model": part_model, "autocorr": part_autocorr, "analyze": part_analyze}[a.part]()
