"""Observation-interpolation baseline (no model), 2D KS at one domain size: periodic cubic-spline
interpolation of noisy observations on the regular stride network used by the DA runs, scored
against the truth at the same analysis times as a 100-cycle DA run (every 5 snapshots, dt_snap
0.1, cycles 51-100). Writes artifacts/experiments/da2d/interp_baseline_L<L>.json.
Usage: python scripts/exp_obs_interp_baseline.py --L 44 --strides 4 8 16 32
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

from ks_latent.config import KSConfig2D
from ks_latent.solver.ks2d import integrate2d

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=float, default=44.0)
ap.add_argument("--strides", type=int, nargs="+", default=[4, 8, 16, 32])
ap.add_argument("--obs-noise-std", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
NX = int(round(a.L / 44 * 128)); STD = 6.172126667325263
cfg = KSConfig2D(Lx=a.L, Ly=a.L, NX=NX, NY=NX, dt=0.02, snapshot_every=5, spinup_time=200.0, seed=a.seed)
u0 = np.load(f"artifacts/cache/ks2d_spinup_L{a.L:g}_NX{NX}_dt0.02_T200_seed{a.seed}.npy")
U = integrate2d(u0, cfg, n_steps=500 * 5)  # 100 cycles x 5 steps
rng = np.random.default_rng(a.seed + 7)
out = {"L": a.L, "NX": NX, "obs_noise_std_field": a.obs_noise_std * STD, "field_std": float(U.std()), "strides": {}}
for s in a.strides:
    errs = []
    for c in range(51, 101):
        u = U[c * 5]
        obs = u[::s, ::s] + a.obs_noise_std * STD * rng.standard_normal(u[::s, ::s].shape)
        m = obs.shape[0]; xo = np.arange(-3, m + 3) * s
        ui = RectBivariateSpline(xo, xo, np.pad(obs, 3, mode="wrap"), kx=3, ky=3)(np.arange(NX), np.arange(NX))
        errs.append(float(np.sqrt(np.mean((ui - u) ** 2))))
    out["strides"][str(s)] = {"spacing_units": s * a.L / NX, "n_obs": (NX // s) ** 2,
                              "rmse_interp": float(np.mean(errs)), "rmse_obs_only_at_obs": a.obs_noise_std * STD}
    print(f"L={a.L:g} stride {s} (spacing {s * a.L / NX:.3f}, {(NX // s) ** 2} obs): interpolation RMSE {np.mean(errs):.3f}")
Path(f"artifacts/experiments/da2d/interp_baseline_L{a.L:g}.json").write_text(json.dumps(out, indent=2))
