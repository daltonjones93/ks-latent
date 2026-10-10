"""J17H (2026-10-10): offline best fit of the structurally wrong physics model
    u_t = -a nabla^2 u - b (-nabla^2)^{order/2} u - (g/2)|nabla u|^2
to standard 2D KS, on an INDEPENDENT truth (seed 3, never assimilated). Objective: rms one-cycle forecast
error (start from the true state, run the model one DA cycle = 0.5 time units, compare). Nelder-Mead on
log(a, b, g). The minimum is the irreducible structural error -- no parameter setting can go below it.
Writes artifacts/experiments/da2d/j17h_fit_order<p>.json."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import scipy.fft as sfft
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from exp_i11_pde_letkf import step_batch  # noqa: E402
from ks_latent.config import KSConfig2D  # noqa: E402
from ks_latent.solver.ks2d import hyper_operator2d, integrate2d  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--order", type=float, default=6.0)
ap.add_argument("--n-starts", type=int, default=40)
ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args()
if a.seed in (0, 1, 2):
    raise ValueError("seeds 0-2 are the DA truths")
cfg = KSConfig2D(Lx=44., Ly=44., NX=128, NY=128, dt=0.02, snapshot_every=5, spinup_time=200., seed=a.seed)
u0 = np.load(f"artifacts/cache/ks2d_spinup_L44_NX128_dt0.02_T200_seed{a.seed}.npy")
n_cyc = 25
U = integrate2d(u0, cfg, n_steps=(a.n_starts + 1) * n_cyc, snapshot_every=n_cyc)
S, T = U[:a.n_starts], U[1:a.n_starts + 1]
V0 = sfft.fft2(S - S.mean(axis=(1, 2), keepdims=True))
hist = []


def err(p):
    A, B, G = np.exp(p)
    op = hyper_operator2d(cfg, A, B, a.order)
    V = V0.copy()
    for _ in range(n_cyc):
        V = step_batch(V, op, nl_coef=G)
    e = float(np.sqrt(np.mean((sfft.ifft2(V).real - T) ** 2)))
    if not np.isfinite(e):
        e = 1e6
    hist.append((A, B, G, e))
    return e


t0 = time.time()
e_true4 = err(np.log([1.0, 1.0, 1.0])) if a.order == 4 else None
# start: match the KS growth-rate peak (K^2 = 1/2, rate 1/4) with b K^order: a = ..., use a = 1, b chosen
# so that a K^2 - b K^p has the same zero crossing as KS (K = 1): b = a
best = None
for x0 in ([1.0, 1.0, 1.0], [1.0, 0.5, 1.0], [0.8, 1.0, 1.0], [1.2, 2.0, 1.0]):
    r = minimize(err, np.log(x0), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 300})
    if best is None or r.fun < best.fun:
        best = r
A, B, G = np.exp(best.x)
out = {"order": a.order, "a": A, "b": B, "g": G, "min_one_cycle_err": best.fun, "ks_true_params_err": e_true4,
       "n_evals": len(hist), "wall_s": time.time() - t0,
       "err_at_a1_b1_g1": err(np.log([1.0, 1.0, 1.0]))}
Path(f"artifacts/experiments/da2d/j17h_fit_order{a.order:g}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
