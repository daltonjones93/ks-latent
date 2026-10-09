"""Task I22 (2026-10-09): estimate the latent model error of one DA cycle from data.

Residual r = E(u(t + n dt)) - M^n(E(u(t))) on an INDEPENDENT 2D KS truth (seed 3, never used by
the DA runs, which use seeds 0-2), at 44^2, unshifted tiling, n = n_prop_steps (one DA cycle).
Pools over sites and times (ks_latent/da/model_error.py). Also reports the one-step residual and
compares the cycle residual's size with the isotropic noise the DA runs use (0.02 per step,
i.e. 0.02 * sqrt(n) accumulated, ignoring growth).

Writes artifacts/experiments/da2d/model_error_<tag>_np<n>.npz (bias, Q, ...) and a .json summary.
Usage: python scripts/exp_i22_model_error.py [--tag p7b_ks2d] [--n-prop-steps 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ks_latent.config import KSConfig2D
from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.da.model_error import estimate_site_model_error
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.ks2d import integrate2d, spinup2d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p7b_ks2d")
    ap.add_argument("--n-prop-steps", type=int, default=5)
    ap.add_argument("--L", type=float, default=44.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n-starts", type=int, default=400, help="residual samples in time (x sites)")
    a = ap.parse_args()
    if a.seed in (0, 1, 2):
        raise ValueError("seeds 0-2 are the DA truths; estimate model error on an independent one")

    AE = f"artifacts/stage1_ae_kolmogorov_crop_{a.tag}.pt"
    PR = f"artifacts/stage2_prop_kolmogorov_crop_{a.tag}.pt"
    ck = torch.load(AE, map_location="cpu", weights_only=False)
    patch = ck["ae_config"].NX // ck["ae_config"].n_sites_x
    g = deployment_grid_2d(a.L, training_dx(ck["train_dataset"]), patch)
    NX, ns = g.NX, g.n_sites
    ae = load_autoencoder_checkpoint_resized_2d(AE, new_n_sites_x=ns, new_n_sites_y=ns, new_NX=NX, new_NY=NX)[0].eval()
    prop = load_propagator_checkpoint_resized_2d(PR, new_n_sites_x=ns, new_n_sites_y=ns)[0].eval()
    std = float(ck["normalization_std"])

    dt = 0.02
    cfg = KSConfig2D(Lx=g.L, Ly=g.L, NX=NX, NY=NX, dt=dt, snapshot_every=round(0.1 / dt), spinup_time=200.0,
                     seed=a.seed)
    cache = Path(f"artifacts/cache/ks2d_spinup_L{g.L:g}_NX{NX}_dt{dt:g}_T200_seed{a.seed}.npy")
    if cache.exists():
        u0 = np.load(cache)
    else:
        u0 = spinup2d(cfg, np.random.default_rng(a.seed + 500))
        np.save(cache, u0)
    n = a.n_prop_steps
    # starts spaced by one cycle; n_starts cycles + n snapshots
    U = integrate2d(u0, cfg, n_steps=(a.n_starts * n + n) * cfg.snapshot_every)  # (S, NX, NX)
    X = torch.tensor((U / std).reshape(U.shape[0], -1), dtype=torch.float32)
    with torch.no_grad():
        Z = torch.cat([ae.encode(X[i:i + 64]) for i in range(0, len(X), 64)])
        starts = torch.arange(0, a.n_starts * n, n)
        zc = Z[starts]
        zp = zc.clone()
        z1 = None
        for k in range(n):
            zn = prop.step(zp, zc)
            zp, zc = zc, zn
            if k == 0:
                z1 = zn
        R_cycle = (Z[starts + n] - zc).numpy().astype(np.float64)
        R_step = (Z[starts + 1] - z1).numpy().astype(np.float64)
    d = Z.shape[1]
    c = d // (ns * ns)
    est = estimate_site_model_error(R_cycle, ns, ns, c)
    est1 = estimate_site_model_error(R_step, ns, ns, c)
    zstd = float(Z.std(0).mean())
    summ = {
        "tag": a.tag, "n_prop_steps": n, "seed": a.seed, "n_samples": est.n_samples, "channels": c,
        "cycle": {"rms": float(np.sqrt(np.trace(est.Q) / c)), "bias_rms": float(np.sqrt(np.mean(est.bias ** 2))),
                  "neighbour_corr": est.neighbour_corr, "Q_eigs": np.linalg.eigvalsh(est.Q).tolist()},
        "step": {"rms": float(np.sqrt(np.trace(est1.Q) / c)), "bias_rms": float(np.sqrt(np.mean(est1.bias ** 2))),
                 "neighbour_corr": est1.neighbour_corr},
        "isotropic_noise_accumulated_rms": 0.02 * np.sqrt(n),
        "latent_climatological_std": zstd,
    }
    out = Path(f"artifacts/experiments/da2d/model_error_{a.tag}_np{n}")
    np.savez(out.with_suffix(".npz"), bias=est.bias, Q=est.Q, bias_step=est1.bias, Q_step=est1.Q, channels=c)
    out.with_suffix(".json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
