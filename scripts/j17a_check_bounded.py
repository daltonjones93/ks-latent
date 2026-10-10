"""J17A A1 (2026-10-10): is u_t = KS(u) - beta P[u'^3] chaotic and bounded at 44^2 over 2000 time units?
Reports field std / max |u| per 100-unit window and the final-state divergence of two nearby starts."""
import json, sys
import numpy as np
from ks_latent.config import KSConfig2D
from ks_latent.solver.ks2d import integrate2d
beta = float(sys.argv[1])
cfg = KSConfig2D(Lx=44., Ly=44., NX=128, NY=128, dt=0.02, snapshot_every=5, spinup_time=200., seed=3)
u0 = np.load("artifacts/cache/ks2d_spinup_L44_NX128_dt0.02_T200_seed3.npy")
U = integrate2d(u0, cfg, n_steps=100000, snapshot_every=500, cubic=beta)  # 2000 units, every 10
W = U[1:].reshape(20, 10, 128, 128)
stats = [{"t_end": 100 * (i + 1), "std": float(w.std()), "max": float(np.abs(w).max())} for i, w in enumerate(W)]
# chaos: two starts 1e-8 apart, separation after 100 units
u1 = U[100]; u2 = u1 + 1e-8 * np.random.default_rng(0).standard_normal(u1.shape)
A = integrate2d(u1, cfg, 5000, snapshot_every=250, cubic=beta); B = integrate2d(u2, cfg, 5000, snapshot_every=250, cubic=beta)
sep = [float(np.sqrt(np.mean((a - b) ** 2))) for a, b in zip(A, B)]
out = {"beta": beta, "windows": stats, "separation_every5": sep, "std_ref_standard_KS": 3.7455}
open(f"artifacts/experiments/da2d/j17a_bounded_beta{beta:g}.json", "w").write(json.dumps(out, indent=1))
print(json.dumps(out))
