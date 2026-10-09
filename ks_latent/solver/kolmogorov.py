"""ETDRK4 spectral solver for forced 2D incompressible Navier-Stokes in
vorticity-streamfunction form -- "Kolmogorov flow" (added 2026-09-28,
`docs/ks_local_field_ltransfer.tex` "Staged next steps" Step 2).

    omega_t + u . grad(omega) = nu * nabla^2 omega + f(y),
    u = (psi_y, -psi_x),   nabla^2 psi = -omega,
    (x,y) in [0,Lx) x [0,Ly), periodic, vorticity omega the sole evolved
    field (velocity/streamfunction are diagnostic, recomputed each
    nonlinear-term evaluation via an exact per-mode Fourier Poisson
    solve -- one division per mode, not an iterative/implicit solve,
    which is why this is barely more expensive than 2D KS despite being
    genuine forced Navier-Stokes).

CPU, float64/complex128 only, same device policy as `ks_latent.solver.ks`/
`ks2d` (brief §1.2: never put the solver on MPS). Uses `scipy.fft`.

**Reuse from `ks_latent.solver.ks2d`, not reimplementation.** The linear
operator here is pure diffusion, `Lhat(kx,ky) = -nu*(kx^2+ky^2)` -- a
strict special case of `ks2d.linear_operator2d`'s general "some function
of K^2" shape, but with no zero-crossing (unlike KS's `K^2-K^4`, which is
positive for `|K|<1` and negative above, `Lhat` here is negative
everywhere except the exact k=0 mode) -- so the SAME contour-integral
ETDRK4 coefficient construction applies unchanged and `etdrk4_coefficients_2d`/
`wavenumbers2d` are imported directly from `ks2d.py`, not reimplemented.
Only the nonlinear-term evaluation and the base/forcing setup are new.

**The forcing and its exact laminar solution.** `f(y) = -nu*forcing_n**3*
cos(forcing_n*y)` is chosen so that `u_lam = (sin(forcing_n*y), 0)`,
`omega_lam = -forcing_n*cos(forcing_n*y)` is an EXACT steady solution of
the full nonlinear equation, not merely of some linearization: `u_lam`/
`omega_lam` depend only on `y`, and `v_lam=0`, so the convective term
`u.grad(omega_lam) = u_lam*d(omega_lam)/dx + v_lam*d(omega_lam)/dy`
vanishes identically (both factors are individually zero -- the first
because `omega_lam` has no x-dependence, the second because `v_lam=0`).
What remains, `0 = nu*nabla^2(omega_lam) + f(y)`, is satisfied by
construction since `nabla^2(omega_lam) = -forcing_n^2*omega_lam` and
`f(y) = -nu*forcing_n^2*omega_lam` (using `omega_lam=-forcing_n*cos(forcing_n*y)`
directly). See `test_base_flow_is_exact_fixed_point` (`tests/unit/
test_solver_kolmogorov.py`) for the numerical confirmation that this
survives discretization AND the ETDRK4 time-stepping exactly (to
float64 roundoff, over many steps) -- a strong, external-literature-free
correctness check that does not depend on reproducing any specific
published critical-Reynolds-number value.

**Why translation equivariance only holds in x, not y -- a genuine
physics difference from both KS variants, not a bug.** `f(y)` explicitly
breaks translation invariance in `y` (2D KS's own nonlinearity has no
explicit spatial dependence at all, so it is equivariant in both axes).
The x-momentum forcing DOES have a residual DISCRETE symmetry: shifting
`y` by any integer multiple of `2*pi/forcing_n` leaves `cos(forcing_n*y)`
invariant, so the equation (and hence `integrate_kolmogorov`) is exactly
equivariant under those specific shifts, but NOT under an arbitrary `y`
shift. `tests/unit/test_solver_kolmogorov.py` tests exactly this
distinction (full equivariance in x, period-`2*pi/forcing_n` equivariance
in y, and explicitly does NOT assert equivariance under an arbitrary y
shift) rather than skipping the y-axis test outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.fft import fft2, ifft2

from ks_latent.config import KolmogorovConfig
from ks_latent.solver.ks2d import ETDRK4Coeffs2D, etdrk4_coefficients_2d, wavenumbers2d


def _dealias_mask(NX: int, NY: int) -> np.ndarray:
    """Orszag 2/3-rule dealiasing mask, `(NX,NY)`, 1.0 within the retained
    band and 0.0 outside (`|mode index| <= (NX or NY)//2 * 2/3`).

    **A real correctness requirement for this system specifically, not a
    precautionary extra.** Both KS solvers in this codebase have never
    needed explicit dealiasing: their own `-k^4` hyperdissipation damps
    high-wavenumber content strongly enough (`Lhat -> -infinity` as
    `k -> infinity`) that aliasing error from the quadratic nonlinearity
    is negligible in practice. Kolmogorov flow's dissipation is only
    `-nu*k^2` -- far weaker at high `k` for the Re values that make the
    laminar solution unstable -- so the same quadratic-nonlinearity
    aliasing (the advection term broadens the spectrum; without
    truncation the top third wraps back into the retained band as
    spurious energy) is not automatically suppressed. This was caught
    empirically, not assumed: an early check found genuine-looking chaos
    (`n_positive=2`) at `NX=NY=48` that vanished (`n_positive=0`, strongly
    stable) at the SAME `nu` when re-measured at production resolution
    `NX=NY=128` -- exactly the signature of resolution-dependent
    numerical noise being mistaken for physical chaos, confirmed once
    this mask made the two resolutions agree (see
    `docs/RESULTS.md`'s "Step 2" progress note for the full comparison)."""
    kx_idx = np.fft.fftfreq(NX, d=1.0 / NX)
    ky_idx = np.fft.fftfreq(NY, d=1.0 / NY)
    KXi, KYi = np.meshgrid(kx_idx, ky_idx, indexing="ij")
    kx_cut = (NX // 2) * (2.0 / 3.0)
    ky_cut = (NY // 2) * (2.0 / 3.0)
    return ((np.abs(KXi) <= kx_cut) & (np.abs(KYi) <= ky_cut)).astype(np.float64)


def linear_operator_kolmogorov(KX: np.ndarray, KY: np.ndarray, nu: float, alpha: float = 0.0) -> np.ndarray:
    """`Lhat(kx,ky) = -nu*(kx^2+ky^2) - alpha`, the Fourier symbol of
    `nu*nabla^2 omega - alpha*omega` (diffusion plus a uniform large-scale
    linear drag, see `KolmogorovConfig`'s own docstring for why `alpha`
    was added). Still pure damping with no zero-crossing except at
    `K=0` (same good-conditioning property as the `alpha=0` case) -- a
    uniform negative shift of an already strictly-negative-except-at-zero
    operator."""
    return -nu * (KX**2 + KY**2) - alpha


def base_flow_vorticity(cfg: KolmogorovConfig, X: np.ndarray | None = None, Y: np.ndarray | None = None) -> np.ndarray:
    """Exact steady-state vorticity `omega_lam = -forcing_n*cos(forcing_n*y)`
    (amplitude-1 laminar solution `u_lam=(sin(forcing_n*y),0)`) -- see
    module docstring for the derivation. Independent of `alpha`: the drag
    term's effect on the balance is folded entirely into `_forcing_fourier`'s
    own `f(y)`, not into the shape of the laminar solution itself. `X`/`Y`
    optionally supplied to avoid recomputing the physical grid at call
    sites that already have it."""
    if X is None or Y is None:
        x = np.arange(cfg.NX) * cfg.Lx / cfg.NX
        y = np.arange(cfg.NY) * cfg.Ly / cfg.NY
        X, Y = np.meshgrid(x, y, indexing="ij")
    return -cfg.forcing_n * np.cos(cfg.forcing_n * Y)


def _forcing_fourier(cfg: KolmogorovConfig) -> np.ndarray:
    """`f_hat = FFT(f(y))`, `f(y) = -(nu*forcing_n**2 + alpha)*forcing_n*
    cos(forcing_n*y)` -- precomputed once (constant in time, depends only
    on `cfg`), cached alongside the rest of the ETDRK4 operator by
    `get_operator_kolmogorov`. `alpha=0` recovers the original
    `-nu*forcing_n**3*cos(forcing_n*y)` exactly (see `KolmogorovConfig`'s
    docstring for the re-derivation with the drag term included)."""
    y = np.arange(cfg.NY) * cfg.Ly / cfg.NY
    x = np.arange(cfg.NX) * cfg.Lx / cfg.NX
    X, Y = np.meshgrid(x, y, indexing="ij")
    f_phys = -(cfg.nu * cfg.forcing_n**2 + cfg.alpha) * cfg.forcing_n * np.cos(cfg.forcing_n * Y)
    return fft2(f_phys)


@dataclass(frozen=True)
class ETDRK4OperatorKolmogorov:
    KX: np.ndarray
    KY: np.ndarray
    Lhat: np.ndarray
    coeffs: ETDRK4Coeffs2D
    f_hat: np.ndarray
    K2_inv: np.ndarray  # 1/(kx^2+ky^2), with the k=0 entry set to 0 (no mean flow)
    dealias: np.ndarray  # 2/3-rule mask, see `_dealias_mask`


@lru_cache(maxsize=32)
def _get_operator_kolmogorov_cached(
    Lx: float, Ly: float, NX: int, NY: int, nu: float, alpha: float, forcing_n: int, dt: float, M: int
) -> ETDRK4OperatorKolmogorov:
    cfg = KolmogorovConfig(Lx=Lx, Ly=Ly, NX=NX, NY=NY, nu=nu, alpha=alpha, forcing_n=forcing_n, dt=dt)
    KX, KY = wavenumbers2d(Lx, Ly, NX, NY)
    Lhat = linear_operator_kolmogorov(KX, KY, nu, alpha)
    coeffs = etdrk4_coefficients_2d(Lhat, dt, M=M)
    K2 = KX**2 + KY**2
    K2_safe = K2.copy()
    K2_safe[0, 0] = 1.0  # avoid 0/0; the k=0 entry of K2_inv is overwritten to 0 next line
    K2_inv = 1.0 / K2_safe
    K2_inv[0, 0] = 0.0
    f_hat = _forcing_fourier(cfg)
    dealias = _dealias_mask(NX, NY)
    return ETDRK4OperatorKolmogorov(
        KX=KX, KY=KY, Lhat=Lhat, coeffs=coeffs, f_hat=f_hat, K2_inv=K2_inv, dealias=dealias
    )


def get_operator_kolmogorov(cfg: KolmogorovConfig, M: int = 32) -> ETDRK4OperatorKolmogorov:
    """ETDRK4 operator for `cfg`, cached by `(Lx, Ly, NX, NY, nu, alpha, forcing_n, dt, M)`."""
    return _get_operator_kolmogorov_cached(
        cfg.Lx, cfg.Ly, cfg.NX, cfg.NY, cfg.nu, cfg.alpha, cfg.forcing_n, cfg.dt, M
    )


def _velocity_from_vorticity(v: np.ndarray, KX: np.ndarray, KY: np.ndarray, K2_inv: np.ndarray):
    """`u = psi_y, v_vel = -psi_x`, `psi_hat = omega_hat * K2_inv` (the
    exact per-mode Fourier Poisson solve, `nabla^2 psi = -omega` =>
    `-K^2*psi_hat = -omega_hat` => `psi_hat = omega_hat/K^2`, k=0 mode
    fixed at 0 -- no mean flow, consistent with the k=0 vorticity mode
    itself being dynamically fixed at 0, see module docstring)."""
    psi_hat = v * K2_inv
    u = ifft2(1j * KY * psi_hat).real
    v_vel = ifft2(-1j * KX * psi_hat).real
    return u, v_vel


def _nonlinear_kolmogorov(v: np.ndarray, op: ETDRK4OperatorKolmogorov) -> np.ndarray:
    """`N(omega_hat) = -FFT(u*omega_x + v_vel*omega_y) + f_hat`. `u`,`v_vel`
    are linear in `omega` (one Poisson solve + one derivative each);
    `omega_x`,`omega_y` are spectral derivatives of `omega` itself -- same
    "product of two linear functionals of the state" bilinear structure
    as both 1D KS's `u*u_x` and 2D KS's `u_x^2+u_y^2`, which is what makes
    the tangent-linear model in `step_with_tangent_kolmogorov` exact and
    cheap (product rule, no new derivation needed beyond linearity).

    **Dealiased (2/3 rule, `op.dealias`), unlike either KS solver** -- see
    `_dealias_mask`'s docstring for why this is a real requirement here.
    `v` is truncated before forming the quadratic product (so the
    genuine-but-aliased top-third content the product would otherwise
    wrap back into the retained band never enters it), and the result is
    truncated again (belt-and-suspenders, standard practice)."""
    KX, KY, K2_inv, mask = op.KX, op.KY, op.K2_inv, op.dealias
    v = v * mask
    u, v_vel = _velocity_from_vorticity(v, KX, KY, K2_inv)
    wx = ifft2(1j * KX * v).real
    wy = ifft2(1j * KY * v).real
    return (-fft2(u * wx + v_vel * wy) + op.f_hat) * mask


def step_kolmogorov(v: np.ndarray, op: ETDRK4OperatorKolmogorov) -> np.ndarray:
    """One ETDRK4 step in Fourier space -- identical algebraic structure
    to `ks2d.step2d`/`ks.step`, only `_nonlinear_kolmogorov` (advection +
    forcing, rather than a KS-style intrinsic nonlinearity) differs.

    Zeros the k=0 (mean vorticity) Fourier mode after every step. Unlike
    2D KS (where this is a genuine gauge-fix removing an undriven,
    physically meaningless degree of freedom), here it is a belt-and-braces
    numerical safeguard: the continuous equation already conserves mean
    vorticity exactly (`f(y)` has zero y-average when `Ly` is an integer
    multiple of the forcing period `2*pi/forcing_n`, and the advective
    term `u.grad(omega) = div(u*omega)` -- using incompressibility
    `div(u)=0` -- integrates to exactly zero over the periodic domain by
    the divergence theorem), so this projection should be a no-op to
    float64 roundoff on a correctly-initialized (zero-mean) trajectory --
    verified directly by `test_mean_vorticity_conservation`, not merely
    asserted."""
    c = op.coeffs
    Nv = _nonlinear_kolmogorov(v, op)
    a = c.E2 * v + c.Q * Nv
    Na = _nonlinear_kolmogorov(a, op)
    b = c.E2 * v + c.Q * Na
    Nb = _nonlinear_kolmogorov(b, op)
    cc = c.E2 * a + c.Q * (2.0 * Nb - Nv)
    Nc = _nonlinear_kolmogorov(cc, op)
    v_next = c.E * v + Nv * c.f1 + 2.0 * (Na + Nb) * c.f2 + Nc * c.f3
    v_next = v_next * op.dealias
    v_next[0, 0] = 0.0
    return v_next


def integrate_kolmogorov(
    u0: np.ndarray,
    cfg: KolmogorovConfig,
    n_steps: int,
    *,
    snapshot_every: int | None = None,
    M: int = 32,
) -> np.ndarray:
    """Integrate vorticity `u0` (`(NX, NY)`) for `n_steps` solver steps of
    size `cfg.dt`. Returns `(n_snapshots + 1, NX, NY)` including the
    initial condition at index 0. `u0`'s own mean is zeroed before the
    first step, same discipline as `ks2d.integrate2d` (see `step_kolmogorov`'s
    docstring: mean vorticity has no forcing/source and should already be
    zero on any physically sensible IC, this just guards against a
    mis-specified one)."""
    if snapshot_every is None:
        snapshot_every = cfg.snapshot_every
    op = get_operator_kolmogorov(cfg, M=M)
    u0 = u0.astype(np.float64) - u0.astype(np.float64).mean()
    v = fft2(u0) * op.dealias
    snapshots = [ifft2(v).real.copy()]
    for step_idx in range(1, n_steps + 1):
        v = step_kolmogorov(v, op)
        if step_idx % snapshot_every == 0:
            snapshots.append(ifft2(v).real.copy())
    return np.stack(snapshots, axis=0)


def default_initial_condition_kolmogorov(cfg: KolmogorovConfig, rng: np.random.Generator) -> np.ndarray:
    """Laminar base flow (`base_flow_vorticity`) plus a small-amplitude
    random low-wavenumber perturbation -- unlike both KS variants' own
    from-scratch small perturbation ICs (which grow via a linear
    instability of the ZERO state), Kolmogorov flow's zero state is
    trivially stable (no forcing acts on `omega=0`... actually it does,
    `f(y)` acts on any state -- but the intended, standard-in-the-literature
    route to the chaotic attractor is to perturb the KNOWN nontrivial
    laminar solution, whose instability is the actual physical mechanism
    generating turbulence here, not a perturbation of the trivial zero
    state)."""
    x = np.arange(cfg.NX) * cfg.Lx / cfg.NX
    y = np.arange(cfg.NY) * cfg.Ly / cfg.NY
    X, Y = np.meshgrid(x, y, indexing="ij")
    u0 = base_flow_vorticity(cfg, X, Y)
    n_modes_per_axis = 4
    amp_scale = 0.05 / n_modes_per_axis
    for j in range(1, n_modes_per_axis + 1):
        for l in range(1, n_modes_per_axis + 1):
            a = rng.normal(scale=amp_scale)
            phi = rng.uniform(0.0, 2.0 * np.pi)
            u0 += a * np.cos(2.0 * np.pi * j * X / cfg.Lx + 2.0 * np.pi * l * Y / cfg.Ly + phi)
    return u0


def step_with_tangent_kolmogorov(
    v: np.ndarray, dV: np.ndarray, op: ETDRK4OperatorKolmogorov
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the base vorticity `v` (`(NX,NY)`) and a batch of tangent
    vectors `dV` (`(k,NX,NY)`, complex, Fourier space) together, for
    `ks_latent.analysis.lyapunov`'s Benettin machinery -- same
    variational-ETDRK4 pattern as `ks2d.step_with_tangent2d`.

    `_nonlinear_kolmogorov`'s advection term is bilinear in `omega`
    (product of two quantities each linear in `omega`: the derived
    velocity `(u,v_vel)`, and the vorticity gradient `(omega_x,omega_y)`),
    so its Frechet derivative follows the product rule exactly, no
    approximation: `D[u.grad(omega)][domega] = du.grad(omega) +
    u.grad(domega)`, where `du` is the velocity derived from `domega` by
    the SAME linear Poisson-solve-then-curl map used for the base
    velocity (`_velocity_from_vorticity` applied to `domega` instead of
    `omega`). The forcing term `f_hat` is constant (state-independent),
    so it contributes nothing to the tangent equation."""
    c = op.coeffs
    KX, KY, K2_inv, mask = op.KX, op.KY, op.K2_inv, op.dealias

    def base_nl(state):
        state = state * mask
        u, v_vel = _velocity_from_vorticity(state, KX, KY, K2_inv)
        wx = ifft2(1j * KX * state).real
        wy = ifft2(1j * KY * state).real
        N = (-fft2(u * wx + v_vel * wy) + op.f_hat) * mask
        return u, v_vel, wx, wy, N

    def tangent_nl(u, v_vel, wx, wy, dstate):
        dstate = dstate * mask[None, :, :]
        du, dv_vel = _velocity_from_vorticity(dstate, KX[None, :, :], KY[None, :, :], K2_inv[None, :, :])
        dwx = ifft2(1j * KX[None, :, :] * dstate, axes=(-2, -1)).real
        dwy = ifft2(1j * KY[None, :, :] * dstate, axes=(-2, -1)).real
        return -fft2(
            du * wx[None, :, :] + u[None, :, :] * dwx + dv_vel * wy[None, :, :] + v_vel[None, :, :] * dwy,
            axes=(-2, -1),
        ) * mask[None, :, :]

    u_v, vv_v, wx_v, wy_v, Nv = base_nl(v)
    dNv = tangent_nl(u_v, vv_v, wx_v, wy_v, dV)

    a = c.E2 * v + c.Q * Nv
    da = c.E2[None, :, :] * dV + c.Q[None, :, :] * dNv
    u_a, vv_a, wx_a, wy_a, Na = base_nl(a)
    dNa = tangent_nl(u_a, vv_a, wx_a, wy_a, da)

    b = c.E2 * v + c.Q * Na
    db = c.E2[None, :, :] * dV + c.Q[None, :, :] * dNa
    u_b, vv_b, wx_b, wy_b, Nb = base_nl(b)
    dNb = tangent_nl(u_b, vv_b, wx_b, wy_b, db)

    cc = c.E2 * a + c.Q * (2.0 * Nb - Nv)
    dcc = c.E2[None, :, :] * da + c.Q[None, :, :] * (2.0 * dNb - dNv)
    u_c, vv_c, wx_c, wy_c, Nc = base_nl(cc)
    dNc = tangent_nl(u_c, vv_c, wx_c, wy_c, dcc)

    v_next = c.E * v + Nv * c.f1 + 2.0 * (Na + Nb) * c.f2 + Nc * c.f3
    dV_next = (
        c.E[None, :, :] * dV
        + dNv * c.f1[None, :, :]
        + 2.0 * (dNa + dNb) * c.f2[None, :, :]
        + dNc * c.f3[None, :, :]
    )
    v_next = v_next * mask
    dV_next = dV_next * mask[None, :, :]
    v_next[0, 0] = 0.0
    dV_next[:, 0, 0] = 0.0
    return v_next, dV_next


def spinup_kolmogorov(cfg: KolmogorovConfig, rng: np.random.Generator, *, M: int = 32) -> np.ndarray:
    """Discard `cfg.spinup_time` of transient (laminar-instability growth
    plus nonlinear saturation onto the turbulent attractor), return the
    resulting `(NX,NY)` vorticity state."""
    u0 = default_initial_condition_kolmogorov(cfg, rng)
    n_spinup_steps = int(round(cfg.spinup_time / cfg.dt))
    if n_spinup_steps == 0:
        return u0
    traj = integrate_kolmogorov(u0, cfg, n_spinup_steps, snapshot_every=n_spinup_steps, M=M)
    return traj[-1]
