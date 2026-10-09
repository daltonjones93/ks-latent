"""Task I11 (2026-10-09): perfect-model ceiling for Part II's 2D KS data assimilation.

A local ETKF (ks_latent/da/letkf_phys.py) whose ensemble members are true 2D KS states,
forecast with the PDE solver itself, assimilating EXACTLY the observations of the latent runs
(exp_h9_tiled_da.py): same truth (cached spinup, same integration), same network (every `stride`
grid points), same noise realization (default_rng(seed + 1), drawn in standardized units and
multiplied by the normalization std). Physical RMSE in field units, steady state = cycles 51-100.

Initial ensemble: truth(0) plus smooth zero-mean random fields with std `--ic-perturb-std`
standardized units (default 0.5; the ensemble must grow its own errors, as the latent one does).
No model error: the only reason this filter is not exact is ensemble size, localization and the
observations. Localization half-width `--loc-c` in physical units (the latent filter's c = 1.5
sites is 8.25 units at 44^2); `--rtps` relaxation to prior spread; `--inflation` multiplicative.

Writes artifacts/experiments/da2d/i11_<out>.json.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import scipy.fft as sfft
import torch

from ks_latent.config import KSConfig2D
from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.da.letkf_phys import LocalETKFPhys2D
from ks_latent.solver.ks2d import _nonlinear2d, get_operator2d, integrate2d, spinup2d
from ks_latent.utils.seeding import set_seed

AE = "artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt"  # only for the grid and normalization


def step_batch(v: np.ndarray, op) -> np.ndarray:
    """`ks2d.step2d` for a batch `(N, NX, NY)` of Fourier states (same ETDRK4 algebra; the k=0
    mode is zeroed per member, which `step2d`'s `v[0, 0] = 0` would get wrong on a batch)."""
    c, KX, KY = op.coeffs, op.KX, op.KY
    Nv = _nonlinear2d(v, KX, KY)
    a = c.E2 * v + c.Q * Nv
    Na = _nonlinear2d(a, KX, KY)
    b = c.E2 * v + c.Q * Na
    Nb = _nonlinear2d(b, KX, KY)
    cc = c.E2 * a + c.Q * (2.0 * Nb - Nv)
    Nc = _nonlinear2d(cc, KX, KY)
    out = c.E * v + Nv * c.f1 + 2.0 * (Na + Nb) * c.f2 + Nc * c.f3
    out[..., 0, 0] = 0.0
    return out


def smooth_field(rng, n, NX, length):
    xi = rng.standard_normal((n, NX, NX))
    f = np.fft.fftfreq(NX)
    kern = np.exp(-2 * (np.pi * length) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2))
    xi = np.fft.ifft2(np.fft.fft2(xi) * kern).real
    xi -= xi.mean(axis=(1, 2), keepdims=True)
    return xi / xi.std(axis=(1, 2), keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=44.0)
    ap.add_argument("--n-ensemble", type=int, default=64)
    ap.add_argument("--n-cycles", type=int, default=100)
    ap.add_argument("--n-prop-steps", type=int, default=5)
    ap.add_argument("--obs-stride", type=int, default=8)
    ap.add_argument("--obs-noise-std", type=float, default=0.1, help="standardized units, as the latent runs")
    ap.add_argument("--loc-c", type=float, default=8.25, help="GC half-width, physical units; <= 0: none")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--inflation", type=float, default=1.0)
    ap.add_argument("--rtps", type=float, default=0.0)
    ap.add_argument("--ic-perturb-std", type=float, default=0.5)
    ap.add_argument("--ic-perturb-len", type=float, default=4.0, help="grid points")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    set_seed(a.seed)
    t0 = time.time()

    ck = torch.load(AE, map_location="cpu", weights_only=False)
    patch = ck["ae_config"].NX // ck["ae_config"].n_sites_x
    g = deployment_grid_2d(a.L, training_dx(ck["train_dataset"]), patch)
    NX = g.NX
    std = float(ck["normalization_std"])

    dt = 0.02
    kcfg = KSConfig2D(Lx=g.L, Ly=g.L, NX=NX, NY=NX, dt=dt, snapshot_every=round(0.1 / dt), spinup_time=200.0,
                      seed=a.seed)
    cache = Path(f"artifacts/cache/ks2d_spinup_L{g.L:g}_NX{NX}_dt{dt:g}_T200_seed{a.seed}.npy")
    if not cache.exists():
        raise FileNotFoundError(f"{cache}: run the latent experiment first so both filters share one truth")
    u0 = np.load(cache)
    n_steps_cycle = a.n_prop_steps * kcfg.snapshot_every
    truth = integrate2d(u0, kcfg, n_steps=n_steps_cycle * a.n_cycles)  # field units, (T+1, NX, NX)
    op = get_operator2d(kcfg)

    s, N = a.obs_stride, a.n_ensemble
    xg = np.arange(0, NX, s)
    obs_xy = np.array([(i, j) for i in xg for j in xg])
    obs_idx = obs_xy[:, 0] * NX + obs_xy[:, 1]
    r_std = a.obs_noise_std * std
    filt = LocalETKFPhys2D(NX, g.L, obs_xy, r_std=r_std, c=a.loc_c if a.loc_c > 0 else None, block=a.block,
                           inflation=a.inflation, rtps=a.rtps)

    rng_ic = np.random.default_rng(a.seed + 7)
    X = truth[0][None] + a.ic_perturb_std * std * smooth_field(rng_ic, N, NX, a.ic_perturb_len)
    X -= X.mean(axis=(1, 2), keepdims=True)
    rng = np.random.default_rng(a.seed + 1)  # same draws as exp_h9_tiled_da.py
    rec = {k: [] for k in ("analysis", "prior", "spread", "obs_pts", "free")}
    Vfree = sfft.fft2(X.mean(0))  # free run: the initial ensemble mean, no assimilation
    with sfft.set_workers(a.workers):
        V = sfft.fft2(X)
        for cyc in range(a.n_cycles):
            for _ in range(n_steps_cycle):
                V = step_batch(V, op)
                Vfree = step_batch(Vfree[None], op)[0]
            Xf = sfft.ifft2(V).real.reshape(N, -1)
            u = truth[(cyc + 1) * n_steps_cycle // kcfg.snapshot_every]
            yg = u[::s, ::s] / std + a.obs_noise_std * rng.standard_normal((NX // s, NX // s))
            y = (yg * std).reshape(-1)
            Xa = filt.analyze(Xf, y, obs_idx)
            if not np.isfinite(Xa).all():
                raise FloatingPointError(f"non-finite analysis at cycle {cyc}")
            f = lambda v: float(np.sqrt(np.mean((v.reshape(NX, NX) - u) ** 2)))  # noqa: E731
            xa = Xa.mean(0)
            rec["analysis"].append(f(xa)); rec["prior"].append(f(Xf.mean(0)))
            rec["spread"].append(float(np.sqrt(np.mean(Xa.var(0, ddof=1)))))
            rec["obs_pts"].append(float(np.sqrt(np.mean((xa[obs_idx] - u.reshape(-1)[obs_idx]) ** 2))))
            rec["free"].append(f(sfft.ifft2(Vfree).real))
            V = sfft.fft2(Xa.reshape(N, NX, NX))
            if cyc % 10 == 9:
                print(f"cycle {cyc + 1}: " + " ".join(f"{k} {v[-1]:.3f}" for k, v in rec.items()), flush=True)
    late = slice(a.n_cycles // 2, None)
    steady = {k: float(np.mean(v[late])) for k, v in rec.items()}
    steady["calib"] = steady["spread"] / steady["analysis"]
    steady["skill"] = steady["free"] / steady["analysis"]
    steady["peak_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
    out = {"args": vars(a), "steady": steady, "per_cycle": rec, "wall_s": time.time() - t0,
           "obs_noise_field_units": r_std}
    Path("artifacts/experiments/da2d").mkdir(parents=True, exist_ok=True)
    Path(f"artifacts/experiments/da2d/i11_{a.out}.json").write_text(json.dumps(out, indent=2))
    print(a.out, {k: round(v, 3) for k, v in steady.items()}, f"{out['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
