"""J17 option 2 (2026-10-10): truth u_t = KS(u) - a u' u'_x at 44^2. For each a: field std, time variability,
growth of a 1e-6 perturbation over 100 units (chaos), and the leading-x/y anisotropy of the spectrum.
The one-cycle error of the plain KS model comes from exp_j16_estimate_pde_model_error.py --truth-aniso."""
import json, sys
import numpy as np
from ks_latent.config import KSConfig2D
from ks_latent.solver.twin import twin_truth
from ks_latent.solver.ks2d import integrate2d
a = float(sys.argv[1])
cfg = KSConfig2D(Lx=44., Ly=44., NX=128, NY=128, dt=0.02, snapshot_every=5, spinup_time=200., seed=3)
u0, kw = twin_truth(cfg, 3, aniso=a)
U = integrate2d(u0, cfg, 10000, snapshot_every=250, **kw)  # 200 units
u2 = U[0] + 1e-6 * np.random.default_rng(0).standard_normal(U[0].shape)
V = integrate2d(u2, cfg, 5000, snapshot_every=250, **kw)
P = np.mean(np.abs(np.fft.fft2(U[1:])) ** 2, axis=0)
kx = np.fft.fftfreq(128)[:, None] ** 2 + 0 * np.fft.fftfreq(128)[None, :]
ky = 0 * np.fft.fftfreq(128)[:, None] + np.fft.fftfreq(128)[None, :] ** 2
out = {"a": a, "std": float(U.std()), "max": float(np.abs(U).max()), "time_std": float(U.std(0).mean()),
       "sep_100u": float(np.sqrt(np.mean((U[20] - V[-1]) ** 2))),
       "aniso_ratio_kx2_over_ky2": float((P * kx).sum() / (P * ky).sum())}
open(f"artifacts/experiments/da2d/j17_aniso_scan_a{a:g}.json", "w").write(json.dumps(out))
print(json.dumps({k: round(v, 4) for k, v in out.items()}))
