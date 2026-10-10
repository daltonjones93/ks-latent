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
from ks_latent.solver.ks2d import aniso_term2d, cubic_term2d, _nonlinear2d, get_operator2d, integrate2d, spinup2d
from ks_latent.utils.seeding import set_seed

AE = "artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt"  # only for the grid and normalization


def step_batch(v: np.ndarray, op, forcing_hat: np.ndarray | None = None, shift: np.ndarray | None = None,
               cubic: float = 0.0, aniso: float = 0.0, nl_coef=1.0) -> np.ndarray:
    """`ks2d.step2d` for a batch `(N, NX, NY)` of Fourier states (same ETDRK4 algebra; the k=0
    mode is zeroed per member, which `step2d`'s `v[0, 0] = 0` would get wrong on a batch)."""
    c, KX, KY = op.coeffs, op.KX, op.KY
    g = np.asarray(nl_coef, dtype=float)
    if g.ndim == 1:  # per-member nonlinearity coefficient (J17H online estimation)
        g = g[:, None, None]
    nlk = (lambda w: _nonlinear2d(w, KX, KY)) if not np.any(g != 1.0) else (lambda w: g * _nonlinear2d(w, KX, KY))
    nl0 = nlk if forcing_hat is None else (lambda w: nlk(w) + forcing_hat)
    cubic = np.asarray(cubic, dtype=float)
    if cubic.ndim == 1:  # per-member coefficients (J17C online estimation): broadcast over the grid
        cubic = cubic[:, None, None]
    nl1 = nl0 if not np.any(cubic) else (lambda w: nl0(w) + cubic_term2d(w, cubic))  # J17A/J17C cubic term
    nl = nl1 if aniso == 0.0 else (lambda w: nl1(w) + aniso_term2d(w, aniso, KX))  # J17 option 2
    Nv = nl(v)
    a = c.E2 * v + c.Q * Nv
    Na = nl(a)
    b = c.E2 * v + c.Q * Na
    Nb = nl(b)
    cc = c.E2 * a + c.Q * (2.0 * Nb - Nv)
    Nc = nl(cc)
    out = c.E * v + Nv * c.f1 + 2.0 * (Na + Nb) * c.f2 + Nc * c.f3
    out[..., 0, 0] = 0.0
    if shift is not None:  # uniform advection, exact as a phase shift (see ks2d.integrate2d)
        out = out * shift
    return out


def spectral_resample(F: np.ndarray, M: int) -> np.ndarray:
    """Resample periodic fields (..., N, N) to (..., M, M) by Fourier truncation (M < N) or
    zero-padding (M > N), keeping the physical amplitude. Exact for band-limited fields."""
    N = F.shape[-1]
    if M == N:
        return F.copy()
    Fh = np.fft.fft2(F)
    k = min(M, N) // 2  # keep |freq| < k: the Nyquist mode of the smaller grid is dropped, so the
    out = np.zeros(F.shape[:-2] + (M, M), dtype=complex)  # resampling is a consistent projection
    idx_src = np.r_[0:k, N - k + 1:N]; idx_dst = np.r_[0:k, M - k + 1:M]
    out[..., idx_dst[:, None], idx_dst[None, :]] = Fh[..., idx_src[:, None], idx_src[None, :]]
    return np.fft.ifft2(out).real * (M / N) ** 2


def smooth_field(rng, n, NX, length):
    xi = rng.standard_normal((n, NX, NX))
    f = np.fft.fftfreq(NX)
    kern = np.exp(-2 * (np.pi * length) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2))
    xi = np.fft.ifft2(np.fft.fft2(xi) * kern).real
    xi -= xi.mean(axis=(1, 2), keepdims=True)
    return xi / xi.std(axis=(1, 2), keepdims=True)


def parse_memory(spec):
    """'eps,tau,kappa[,coupling]' -> MemoryParams (J17B), or None."""
    if not spec:
        return None
    from ks_latent.solver.ks2d_memory import MemoryParams
    f = spec.split(",")
    return MemoryParams(float(f[0]), float(f[1]), float(f[2]), f[3] if len(f) > 3 else "lap")


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
    ap.add_argument("--model-nx", type=int, default=None,
                    help="I28 option 1 (resolution error): the filter's PDE runs on an M x M grid (M < truth NX); "
                         "observations at the same physical points; scored after spectral upsampling to the truth grid.")
    # I28 (user design, 2026-10-09): the TRUTH is the standard system (the one the latent model was
    # trained on); the physics filter's MODEL is wrong. These options change only the filter's model.
    ap.add_argument("--model-lap-coef", type=float, default=1.0, help="filter model: coefficient on nabla^2")
    ap.add_argument("--model-advect-x", type=float, default=0.0, help="filter model: spurious mean-flow advection")
    ap.add_argument("--model-forcing-amp", type=float, default=0.0, help="filter model: spurious terrain forcing")
    # J16 (2026-10-09): standard model-error remedies for the physics filter.
    ap.add_argument("--estimate-param", choices=["none", "lap", "adv", "cub", "abg"], default="none",
                    help="online estimation (state augmentation, global ETKF) of the nabla^2 coefficient ('lap') or "
                         "the mean-flow speed ('adv'); starts from the wrong model value (--model-lap-coef / "
                         "--model-advect-x)")
    ap.add_argument("--param-init-std", type=float, default=0.05,
                    help="initial parameter spread (for 'abg': RELATIVE to each starting value)")
    ap.add_argument("--param-rtps", type=float, default=0.9)
    ap.add_argument("--param-min-std", type=float, default=0.005, help="spread floor (for 'abg': relative)")
    # J17H (2026-10-10, user design): structurally wrong physics model with ALL coefficients estimable,
    #   u_t = -a nabla^2 u - b (-nabla^2)^{order/2} u - (g/2)|nabla u|^2,   order != 4,
    # so no (a, b, g) reproduces KS. '--estimate-param abg' estimates (a, b, g) jointly online.
    ap.add_argument("--model-order", type=float, default=4.0, help="J17H: order of the damping operator (KS: 4)")
    ap.add_argument("--model-hyper-coef", type=float, default=1.0, help="J17H: b, coefficient of the damping operator")
    ap.add_argument("--model-nl-coef", type=float, default=1.0, help="J17H: g, coefficient of the |grad u|^2 term")
    ap.add_argument("--model-error-file", default=None,
                    help="npz from exp_j16_estimate_pde_model_error.py: bias (mean one-cycle residual) and spectral "
                         "shape of the residual, on the model grid")
    ap.add_argument("--bias-correct", action="store_true", help="add the mean one-cycle residual to each forecast")
    ap.add_argument("--q-scale", type=float, default=0.0,
                    help="additive model-error noise: q_scale x residual std, residual spectral shape")
    ap.add_argument("--truth-lap-coef", type=float, default=1.0,
                    help="I28: truth uses coefficient a on nabla^2; the filter's PDE keeps a = 1 (imperfect model).")
    ap.add_argument("--truth-advect-x", type=float, default=0.0, help="I28: truth has mean-flow advection c (x); models do not")
    ap.add_argument("--truth-forcing-amp", type=float, default=0.0, help="I28: truth has terrain-like forcing (std); models do not")
    # J17A (2026-10-10): missing process. Truth u_t = KS(u) - beta P[u'^3]; the filter's KS model lacks it unless
    # --model-cubic-coef is given (the perfect-model floor).
    ap.add_argument("--truth-memory", default=None,
                    help="J17B: 'eps,tau,kappa[,coupling]' -- truth has the hidden memory field v (ks2d_memory.py)")
    ap.add_argument("--model-memory", default=None,
                    help="J17B floor / wrong-tau runs: the filter's model carries its own (u, v) with these "
                         "'eps,tau,kappa[,coupling]'; v is unobserved and updated through ensemble covariances")
    ap.add_argument("--truth-cubic", type=float, default=0.0, help="J17A: truth's cubic damping beta")
    ap.add_argument("--model-cubic-coef", type=float, default=0.0, help="J17A: filter model's cubic beta (floor runs)")
    ap.add_argument("--norm-std", type=float, default=None,
                    help="field std defining the standardized obs noise (default: the p7b checkpoint's, i.e. standard "
                         "KS data); J17A passes the new system's training-data std so both filters see the same noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    set_seed(a.seed)
    t0 = time.time()

    ck = torch.load(AE, map_location="cpu", weights_only=False)
    patch = ck["ae_config"].NX // ck["ae_config"].n_sites_x
    g = deployment_grid_2d(a.L, training_dx(ck["train_dataset"]), patch)
    NX = g.NX
    std = float(ck["normalization_std"]) if a.norm_std is None else a.norm_std

    dt = 0.02
    kcfg = KSConfig2D(Lx=g.L, Ly=g.L, NX=NX, NY=NX, dt=dt, snapshot_every=round(0.1 / dt), spinup_time=200.0,
                      seed=a.seed)
    cache = Path(f"artifacts/cache/ks2d_spinup_L{g.L:g}_NX{NX}_dt{dt:g}_T200_seed{a.seed}.npy")
    if not cache.exists():
        raise FileNotFoundError(f"{cache}: run the latent experiment first so both filters share one truth")
    from ks_latent.solver.twin import twin_truth
    n_steps_cycle = a.n_prop_steps * kcfg.snapshot_every
    if a.truth_memory:  # J17B: hidden memory field; only u is observed and scored
        from ks_latent.solver.ks2d_memory import integrate_memory, memory_truth
        tmp = parse_memory(a.truth_memory)
        u0, v0 = memory_truth(kcfg, a.seed, tmp)
        truth = integrate_memory(u0, v0, kcfg, n_steps_cycle * a.n_cycles, tmp)[0]
    else:
        u0, truth_kw = twin_truth(kcfg, a.seed, a.truth_lap_coef, a.truth_advect_x, a.truth_forcing_amp,
                                 cubic=a.truth_cubic)
        truth = integrate2d(u0, kcfg, n_steps=n_steps_cycle * a.n_cycles, **truth_kw)  # field units, (T+1, NX, NX)
    M = a.model_nx or NX
    if NX % M or a.obs_stride % (NX // M):
        raise ValueError(f"model grid {M} must divide {NX}, and the obs stride {a.obs_stride} must be a multiple of {NX // M}")
    rr = NX // M
    mcfg = KSConfig2D(Lx=g.L, Ly=g.L, NX=M, NY=M, dt=dt, snapshot_every=kcfg.snapshot_every, spinup_time=200.0,
                      seed=a.seed)
    # the filter's model: the standard equation on its own grid, unless a model error is requested
    if a.model_order != 4.0 or a.model_hyper_coef != 1.0:
        from ks_latent.solver.ks2d import hyper_operator2d
        op = hyper_operator2d(mcfg, a.model_lap_coef, a.model_hyper_coef, a.model_order)
    elif a.model_lap_coef != 1.0:
        from ks_latent.solver.ks2d import modified_operator2d
        op = modified_operator2d(mcfg, a.model_lap_coef)
    else:
        op = get_operator2d(mcfg)
    m_shift = None if a.model_advect_x == 0.0 else np.exp(-1j * a.model_advect_x * op.KX * dt)
    m_forc = None
    if a.model_forcing_amp != 0.0:
        from ks_latent.solver.twin import terrain_forcing
        m_forc = np.fft.fft2(terrain_forcing(mcfg, a.model_forcing_amp))

    from types import SimpleNamespace
    from ks_latent.da.param_est import global_param_update, global_params_update
    from ks_latent.solver.ks2d import ETDRK4Coeffs2D, etdrk4_coefficients_2d
    K2m = op.KX ** 2 + op.KY ** 2

    def batch_lap_op(theta):  # per-member operator for member-specific nabla^2 coefficients
        parts = []
        for i in range(0, len(theta), 8):
            Lh = theta[i:i + 8, None, None] * K2m[None] - a.model_hyper_coef * K2m[None] ** (a.model_order / 2)
            parts.append(etdrk4_coefficients_2d(Lh, dt))
        co = ETDRK4Coeffs2D(**{f: np.concatenate([getattr(c, f) for c in parts]) for f in ("E", "E2", "Q", "f1", "f2", "f3")})
        return SimpleNamespace(coeffs=co, KX=op.KX, KY=op.KY)

    def batch_ab_op(A, Bc):  # per-member (a, b) operator for the J17H model
        parts = []
        for i in range(0, len(A), 8):
            Lh = A[i:i + 8, None, None] * K2m[None] - Bc[i:i + 8, None, None] * K2m[None] ** (a.model_order / 2)
            parts.append(etdrk4_coefficients_2d(Lh, dt))
        co = ETDRK4Coeffs2D(**{f: np.concatenate([getattr(c, f) for c in parts]) for f in ("E", "E2", "Q", "f1", "f2", "f3")})
        return SimpleNamespace(coeffs=co, KX=op.KX, KY=op.KY)

    me = None
    if a.model_error_file:
        me = np.load(a.model_error_file)
        if me["bias"].shape != (M, M):
            raise ValueError(f"model-error file is for grid {me['bias'].shape}, model grid is {(M, M)}")
    rng_q = np.random.default_rng(a.seed + 11)

    def model_error_noise(n):
        z = np.fft.fft2(rng_q.standard_normal((n, M, M)))
        f = np.fft.ifft2(z * np.sqrt(me["spec_shape"])[None]).real
        f -= f.mean(axis=(1, 2), keepdims=True)
        return f / f.std(axis=(1, 2), keepdims=True) * float(me["sigma"])

    s, N = a.obs_stride, a.n_ensemble
    xg = np.arange(0, NX, s)  # observed rows/cols on the truth grid
    obs_xy = np.array([(i // rr, j // rr) for i in xg for j in xg])  # same physical points, model grid
    obs_idx = obs_xy[:, 0] * M + obs_xy[:, 1]
    r_std = a.obs_noise_std * std
    filt = LocalETKFPhys2D(M, g.L, obs_xy, r_std=r_std, c=a.loc_c if a.loc_c > 0 else None,
                           block=max(1, a.block // rr), inflation=a.inflation, rtps=a.rtps)

    rng_ic = np.random.default_rng(a.seed + 7)
    X = truth[0][None] + a.ic_perturb_std * std * smooth_field(rng_ic, N, NX, a.ic_perturb_len)
    X -= X.mean(axis=(1, 2), keepdims=True)
    X = spectral_resample(X, M)
    up = lambda v: spectral_resample(v.reshape(M, M), NX)  # noqa: E731  model grid -> truth grid
    rng = np.random.default_rng(a.seed + 1)  # same draws as exp_h9_tiled_da.py
    rec = {k: [] for k in ("analysis", "prior", "spread", "obs_pts", "free", "theta_mean", "theta_std")}
    theta = None
    if a.estimate_param != "none":
        if a.estimate_param == "abg":
            t0v = np.array([a.model_lap_coef, a.model_hyper_coef, a.model_nl_coef])
            theta = t0v * (1 + a.param_init_std * np.random.default_rng(a.seed + 13).standard_normal((N, 3)))
            abg_floor = a.param_min_std * np.abs(t0v)
        else:
            theta0 = {"lap": a.model_lap_coef, "adv": a.model_advect_x, "cub": a.model_cubic_coef}[a.estimate_param]
            theta = theta0 + a.param_init_std * np.random.default_rng(a.seed + 13).standard_normal(N)
    Vfree = sfft.fft2(X.mean(0))  # free run: the initial ensemble mean, no assimilation
    mmop = None
    if a.model_memory:  # J17B: the model carries a hidden v (starts at 0: the filter does not know it)
        if theta is not None or me is not None or M != NX:
            raise ValueError("--model-memory is for the floor / wrong-tau runs: no estimation, bias or coarse grid")
        from ks_latent.solver.ks2d_memory import MemoryOperator, step_memory
        mmop = MemoryOperator(mcfg, parse_memory(a.model_memory), op_u=op)
        Wv, Wvfree = np.zeros((N, M, M), dtype=complex), np.zeros((M, M), dtype=complex)
    with sfft.set_workers(a.workers):
        V = sfft.fft2(X)
        for cyc in range(a.n_cycles):
            op_e, shift_e = op, m_shift
            if theta is not None and a.estimate_param == "lap":
                op_e = batch_lap_op(theta)
            if theta is not None and a.estimate_param == "adv":
                shift_e = np.exp(-1j * theta[:, None, None] * op.KX[None] * dt)
            cub_e = theta if (theta is not None and a.estimate_param == "cub") else a.model_cubic_coef
            g_e = a.model_nl_coef
            if theta is not None and a.estimate_param == "abg":
                op_e, g_e = batch_ab_op(theta[:, 0], theta[:, 1]), theta[:, 2]
            for _ in range(n_steps_cycle):
                if mmop is not None:
                    V, Wv = step_memory(V, Wv, mmop)
                    Vfree, Wvfree = step_memory(Vfree, Wvfree, mmop)
                    continue
                V = step_batch(V, op_e, m_forc, shift_e, cub_e, nl_coef=g_e)
                Vfree = step_batch(Vfree[None], op, m_forc, m_shift, a.model_cubic_coef, nl_coef=a.model_nl_coef)[0]
            Xf = sfft.ifft2(V).real.reshape(N, -1)  # (N, M*M)
            if me is not None:
                if a.bias_correct:
                    Xf = Xf + me["bias"].reshape(1, -1)
                if a.q_scale > 0:
                    Xf = Xf + a.q_scale * model_error_noise(N).reshape(N, -1)
            u = truth[(cyc + 1) * n_steps_cycle // kcfg.snapshot_every]
            yg = u[::s, ::s] / std + a.obs_noise_std * rng.standard_normal((NX // s, NX // s))
            y = (yg * std).reshape(-1)
            if theta is not None:  # parameter update with the PRIOR ensemble (simultaneous with the state)
                Yp = Xf[:, obs_idx] - Xf[:, obs_idx].mean(0)
                if a.estimate_param == "abg":
                    theta = global_params_update(theta, Yp, y - Xf[:, obs_idx].mean(0), r_std,
                                                 rtps=a.param_rtps, min_std=abg_floor)
                    theta[:, 1] = np.maximum(theta[:, 1], 1e-6 * abs(a.model_hyper_coef))  # damping must stay > 0
                    rec["theta_mean"].append(theta.mean(0).tolist()); rec["theta_std"].append(theta.std(0, ddof=1).tolist())
                else:
                    theta = global_param_update(theta, Yp, y - Xf[:, obs_idx].mean(0), r_std,
                                                rtps=a.param_rtps, min_std=a.param_min_std)
                    rec["theta_mean"].append(float(theta.mean())); rec["theta_std"].append(float(theta.std(ddof=1)))
            if mmop is not None:  # augmented state [u; v]: v updated through its covariance with observed u
                XVa = filt.analyze(np.concatenate([Xf, sfft.ifft2(Wv).real.reshape(N, -1)], axis=1), y, obs_idx)
                Xa = XVa[:, :M * M]
                Wv = sfft.fft2(XVa[:, M * M:].reshape(N, M, M))
                Wv[:, 0, 0] = 0.0
            else:
                Xa = filt.analyze(Xf, y, obs_idx)
            if not np.isfinite(Xa).all():
                raise FloatingPointError(f"non-finite analysis at cycle {cyc}")
            f = lambda v: float(np.sqrt(np.mean((up(v) - u) ** 2)))  # noqa: E731
            xa = Xa.mean(0)
            rec["analysis"].append(f(xa)); rec["prior"].append(f(Xf.mean(0)))
            rec["spread"].append(float(np.sqrt(np.mean(Xa.var(0, ddof=1)))))
            rec["obs_pts"].append(float(np.sqrt(np.mean((up(xa)[::s, ::s] - u[::s, ::s]) ** 2))))
            rec["free"].append(f(sfft.ifft2(Vfree).real))
            V = sfft.fft2(Xa.reshape(N, M, M))
            if cyc % 10 == 9:
                print(f"cycle {cyc + 1}: " + " ".join(f"{k} {v[-1]:.3f}" for k, v in rec.items() if v), flush=True)
    late = slice(a.n_cycles // 2, None)
    steady = {k: float(np.mean(v[late])) for k, v in rec.items() if v and not isinstance(v[0], list)}
    vec = {k: np.mean(np.array(v[late]), axis=0).tolist() for k, v in rec.items() if v and isinstance(v[0], list)}
    steady["calib"] = steady["spread"] / steady["analysis"]
    steady["skill"] = steady["free"] / steady["analysis"]
    steady["peak_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
    out = {"args": vars(a), "steady": steady, "steady_vec": vec, "per_cycle": rec, "wall_s": time.time() - t0,
           "obs_noise_field_units": r_std}
    Path("artifacts/experiments/da2d").mkdir(parents=True, exist_ok=True)
    Path(f"artifacts/experiments/da2d/i11_{a.out}.json").write_text(json.dumps(out, indent=2))
    print(a.out, {k: round(v, 3) for k, v in steady.items()}, {k: [round(x, 4) for x in v] for k, v in vec.items()},
          f"{out['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
