"""J17S screening (2026-10-10, user-directed: "write and run the screening for candidates 1 and 2").

Candidate truths, both Markov in u, local and translation invariant, added to standard 2D KS at 44^2:
  ganiso (candidate 1): -(delta/2) P[u_x^2 - u_y^2]   (anisotropic KS; `ks2d.grad_aniso_term2d`)
  aniso  (candidate 2): -a u' u'_x                    (Burgers-type steepening in x; `ks2d.aniso_term2d`)

For one (kind, level), on an INDEPENDENT truth (seed 3, never assimilated):
 1. truth statistics over 200 time units: field std, max |u|, time variability, x/y anisotropy of the
    gradient energy, boundedness; chaos check: rms separation of a 1e-6 perturbation after 100 units;
 2. one-cycle (0.5 time units) forecast error of the physics model
        u_t = -a lap u - b lap^2 u - (g/2)|grad u|^2 - c . grad u
    (the parameters the physics filter can estimate online: a, b, g and uniform advection c), started
    from the true state: at the KS values (1, 1, 1, 0) and at the best fit (Nelder-Mead on the rms
    error). The best-fit error is the irreducible structural error. Each error is split into bias
    (per-pixel mean over starts) and the state-dependent part (std of the rest), as in
    exp_j16_estimate_pde_model_error.py; ratios use sigma_obs = 0.1 x the truth's field std.
Writes artifacts/experiments/da2d/j17s_screen_<kind><level>.json. Pre-registered target (J17S in
docs/full_paper_tasks.md): state-dependent best-fit error of ~0.1, ~0.2 and ~0.35 sigma_obs at three
levels that stay bounded and chaotic."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import scipy.fft as sfft
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from exp_i11_pde_letkf import step_batch  # noqa: E402
from ks_latent.config import KSConfig2D  # noqa: E402
from ks_latent.solver.ks2d import hyper_operator2d, integrate2d  # noqa: E402
from ks_latent.solver.twin import twin_truth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["ganiso", "aniso"], required=True)
ap.add_argument("--level", type=float, required=True)
ap.add_argument("--n-starts", type=int, default=40)
ap.add_argument("--start-spacing", type=float, default=1.0, help="time units between forecast starts")
ap.add_argument("--maxiter", type=int, default=700)
ap.add_argument("--no-fit", action="store_true")
a = ap.parse_args()
SEED = 3
cfg = KSConfig2D(Lx=44., Ly=44., NX=128, NY=128, dt=0.02, snapshot_every=5, spinup_time=200., seed=SEED)
t0 = time.time()
u0, kw = twin_truth(cfg, SEED, **({a.kind: a.level} if a.level != 0 else {}))

# 1. statistics and chaos
U = integrate2d(u0, cfg, 10000, snapshot_every=25, **kw)  # 200 units, every 0.5
if not np.all(np.isfinite(U)) or np.abs(U).max() > 1e3:
    raise RuntimeError(f"truth unbounded: max |u| = {np.abs(U).max()}")
V = integrate2d(U[0] + 1e-6 * np.random.default_rng(0).standard_normal(U[0].shape), cfg, 5000,
                snapshot_every=5000, **kw)
std = float(U.std())
kx = 2 * np.pi * np.fft.fftfreq(cfg.NX, d=cfg.Lx / cfg.NX)
P = np.mean(np.abs(sfft.fft2(U[1::4])) ** 2, axis=0)
gx, gy = float((P * kx[:, None] ** 2).sum()), float((P * kx[None, :] ** 2).sum())
stats = {"field_std": std, "max_abs": float(np.abs(U).max()), "time_std": float(U.std(0).mean()),
         "sep_100u_over_std": float(np.sqrt(np.mean((U[200] - V[-1]) ** 2)) / std),
         "grad_energy_x_over_y": gx / gy, "sigma_obs": 0.1 * std}
print(f"[{a.kind}{a.level:g}] stats {json.dumps({k: round(v, 4) for k, v in stats.items()})} "
      f"[{time.time() - t0:.0f}s]", flush=True)

# 2. one-cycle error of the physics model
n_cyc = 25
gap = int(round(a.start_spacing / cfg.dt))
S = integrate2d(U[-1], cfg, a.n_starts * gap, snapshot_every=gap, **kw)[:a.n_starts]
T = np.stack([integrate2d(s, cfg, n_cyc, snapshot_every=n_cyc, **kw)[-1] for s in S])
V0 = sfft.fft2(S - S.mean(axis=(1, 2), keepdims=True))
op_cache = {}


def forecast(A, B, G, cx, cy):
    op = hyper_operator2d(cfg, A, B, 4.0)
    shift = np.exp(-1j * (cx * op.KX + cy * op.KY) * cfg.dt)
    Vf = V0.copy()
    for _ in range(n_cyc):
        Vf = step_batch(Vf, op, shift=shift, nl_coef=G)
    return sfft.ifft2(Vf).real


def split(R):
    bias = R.mean(0)
    return {"rms": float(np.sqrt(np.mean(R ** 2))), "bias_rms": float(np.sqrt(np.mean(bias ** 2))),
            "state": float((R - bias).std())}


n_eval = [0]


def obj(p):
    n_eval[0] += 1
    R = forecast(np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3], p[4]) - T
    e = float(np.sqrt(np.mean(R ** 2)))
    return e if np.isfinite(e) else 1e6


out = {"kind": a.kind, "level": a.level, "seed": SEED, "n_starts": a.n_starts, "stats": stats,
       "ks_params": split(forecast(1., 1., 1., 0., 0.) - T)}
print(f"[{a.kind}{a.level:g}] KS params {out['ks_params']} [{time.time() - t0:.0f}s]", flush=True)
if not a.no_fit:
    best = None
    for x0 in ([0., 0., 0., 0., 0.], [0.1, 0.1, 0.05, 0.1, 0.]):
        simplex = np.array([x0] + [np.array(x0) + d for d in 0.1 * np.eye(5)])
        r = minimize(obj, x0, method="Nelder-Mead",
                     options={"initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-5, "maxiter": a.maxiter})
        if best is None or r.fun < best.fun:
            best = r
    A, B, G = np.exp(best.x[:3])
    cx, cy = best.x[3:]
    out["fit"] = {"a": A, "b": B, "g": G, "cx": cx, "cy": cy, "converged": bool(best.success),
                  **split(forecast(A, B, G, cx, cy) - T)}
    out["n_evals"] = n_eval[0]
    for k in ("ks_params", "fit"):
        out[k]["state_over_sigma_obs"] = out[k]["state"] / stats["sigma_obs"]
out["wall_s"] = time.time() - t0
Path(f"artifacts/experiments/da2d/j17s_screen_{a.kind}{a.level:g}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out), flush=True)
