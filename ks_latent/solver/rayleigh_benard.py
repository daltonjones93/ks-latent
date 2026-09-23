"""2D Boussinesq Rayleigh-Benard convection, vorticity-streamfunction
pseudospectral solver. CPU, float64 only (brief's own solver-device
policy: data generation is offline and correctness-critical; MPS lacks
float64 anyway -- never put this solver on MPS).

See `ks_latent.config.RayleighBenardConfig`'s docstring for the exact
governing equations,

    (1/Pr) (d(omega)/dt + u . grad(omega)) = laplacian(omega) + Ra * d(theta)/dx
    d(theta)/dt + u . grad(theta) = laplacian(theta) + w
    laplacian(psi) = -omega,   u = d(psi)/dz,   w = -d(psi)/dx

the free-slip/isothermal boundary conditions (psi=omega=theta=0 at
z=0,1), and why the box's aspect ratio is chosen so the exact textbook
critical Rayleigh number `Ra_c = 27*pi**4/4` is reachable.

**Spectral representation.** Every field is stored as a REAL sine series
in z (`f(x,z) = sum_m f_m(x) sin(m*pi*z)`, m=1..Nz) composed with a
`scipy.fft.rfft` in the periodic x direction -- shape `(Nz, Nx//2+1)`
complex128. Both `laplacian` and the z-independent x-derivative are
DIAGONAL in this representation (sin(m*pi*z) is an eigenfunction of
d^2/dz^2 with eigenvalue `-(m*pi)**2`, by construction of the boundary
conditions), which is exactly why this basis was chosen.

**z-derivatives that change parity** (e.g. d(psi)/dz, needed for the
velocity u) are NOT computed by chaining `scipy.fft.dst`/`dct` calls --
matching scipy's several DST/DCT *type* conventions (each tied to a
different grid-point-inclusion convention) to this class's own DST-I
interior grid (`z_j = j/(Nz+1)`) is a well-known source of off-by-one/
wrong-type bugs in spectral codes. Instead this module builds two small
explicit `(Nz, Nz)` matrices ONCE (`_sine_cosine_matrices`, `lru_cache`'d
per `Nz`):

  - `S[j, m] = sqrt(2/(Nz+1)) * sin(m*pi*z_j)` -- the DST-I basis,
    provably ORTHONORMAL (`S.T @ S = I`, a standard discrete-sine-
    transform identity, verified directly in `tests/unit/
    test_rayleigh_benard.py::test_sine_cosine_matrices_orthonormal_and_derivative`),
    so `S` synthesizes physical values from sine coefficients and `S.T`
    (== `S`'s exact inverse) analyzes physical values back into sine
    coefficients.
  - `C[j, m] = sqrt(2/(Nz+1)) * cos(m*pi*z_j)` -- used ONLY to SYNTHESIZE
    (coefficients -> physical z-derivative values), via
    `C @ (m*pi * coeffs)`; never inverted, so its own orthonormality
    (which does not hold on this grid) is irrelevant.

At `Nz=64` these are 64x64 matmuls, applied across every x-frequency at
once -- negligible cost next to the `rfft`/`irfft` calls, and removes an
entire class of transform-convention bugs at essentially no performance
cost for this project's target resolution.

**Nonlinear (advection) term**, pseudospectral: transform u, w, and the
needed derivatives of omega/theta to physical `(Nz, Nx)` space (2/3-rule
dealiased first), multiply pointwise, transform the product back to sine-
coefficient space. `u*d(omega)/dx + w*d(omega)/dz` is provably a pure
sine series in z again (cos(z)*sin(z) and sin(z)*cos(z) products are both
pure sine, by the standard product-to-sum identities), so the forward
transform back uses `S.T` exactly as for any other sine-represented
field -- self-consistent with the state representation.

**Time integration**: CNAB2 (Crank-Nicolson for the diagonal diffusion
operator, 2nd-order Adams-Bashforth for the nonlinear+buoyancy/background-
advection terms treated explicitly) -- the standard IMEX choice for this
class of problem: since diffusion is diagonal here, the "implicit solve"
is a trivial elementwise divide, no linear system ever needs to be
assembled. Timestep is adaptive (CFL-limited on `u`, `w`), the last
sub-step before each snapshot boundary is shrunk to land on it exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from ks_latent.config import RayleighBenardConfig


@lru_cache(maxsize=8)
def _sine_cosine_matrices(Nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns `(S, C, m_pi, z_grid)`.

    `S`: (Nz, Nz) orthonormal DST-I synthesis/analysis matrix.
    `C`: (Nz, Nz) matching cosine-synthesis-only matrix, for z-derivatives.
    `m_pi`: (Nz,) mode angular frequencies `m*pi`, m=1..Nz.
    `z_grid`: (Nz,) physical interior grid points `j/(Nz+1)`, j=1..Nz.
    """
    j = np.arange(1, Nz + 1, dtype=np.float64)
    m = np.arange(1, Nz + 1, dtype=np.float64)
    z_grid = j / (Nz + 1)
    # S[j_idx, m_idx] = sqrt(2/(Nz+1)) * sin(m*pi*z_j)
    S = np.sqrt(2.0 / (Nz + 1)) * np.sin(np.outer(z_grid, m) * np.pi)
    C = np.sqrt(2.0 / (Nz + 1)) * np.cos(np.outer(z_grid, m) * np.pi)
    m_pi = m * np.pi
    return S, C, m_pi, z_grid


def wavenumbers_x(cfg: RayleighBenardConfig) -> np.ndarray:
    """Angular x-wavenumbers for the rfft representation, shape (Nx//2+1,)."""
    return 2.0 * np.pi * rfftfreq(cfg.Nx, d=cfg.Lx / cfg.Nx)


@dataclass(frozen=True)
class _Operators:
    """Cached per-(cfg.Nz, cfg.Nx, cfg.Lx, cfg.dealias_fraction,
    cfg.fft_workers) arrays, built once per config."""

    S: np.ndarray  # (Nz, Nz)
    C: np.ndarray  # (Nz, Nz)
    m_pi: np.ndarray  # (Nz,)
    kx: np.ndarray  # (Nkx,)
    neg_lap: np.ndarray  # (Nz, Nkx) = kx**2 + (m*pi)**2, i.e. laplacian symbol is -neg_lap
    dealias_mask: np.ndarray  # (Nz, Nkx) bool
    fft_workers: int  # passed to every scipy.fft call below (multi-core CPU vectorization)


@lru_cache(maxsize=8)
def _get_operators(Nz: int, Nx: int, Lx: float, dealias_fraction: float, fft_workers: int) -> _Operators:
    S, C, m_pi, _ = _sine_cosine_matrices(Nz)
    kx = 2.0 * np.pi * rfftfreq(Nx, d=Lx / Nx)
    neg_lap = kx[None, :] ** 2 + m_pi[:, None] ** 2
    kx_cut = dealias_fraction * kx.max()
    m_cut = dealias_fraction * m_pi.max()
    dealias_mask = (kx[None, :] <= kx_cut) & (m_pi[:, None] <= m_cut)
    return _Operators(
        S=S, C=C, m_pi=m_pi, kx=kx, neg_lap=neg_lap, dealias_mask=dealias_mask, fft_workers=fft_workers
    )


def get_operators(cfg: RayleighBenardConfig) -> _Operators:
    return _get_operators(cfg.Nz, cfg.Nx, cfg.Lx, cfg.dealias_fraction, cfg.fft_workers)


def physical_from_sine(coef: np.ndarray, ops: _Operators) -> np.ndarray:
    """Sine coefficients (Nz, Nkx) complex -> physical (Nz, Nx) real.
    Relies on scipy's default `irfft` output-length inference
    (`n = 2*(Nkx-1)`), which is exact for even `Nx` -- this project's
    `RayleighBenardConfig.Nx` default (64) and every value used so far."""
    z_phys = ops.S @ coef  # (Nz, Nkx), still complex/rfft in x
    return irfft(z_phys, axis=-1, workers=ops.fft_workers)


def sine_from_physical(vals_x_phys: np.ndarray, ops: _Operators) -> np.ndarray:
    """Physical (Nz, Nx) real -> sine coefficients (Nz, Nkx) complex."""
    x_spec = rfft(vals_x_phys, axis=-1, workers=ops.fft_workers)  # (Nz, Nkx)
    return ops.S.T @ x_spec


def dz_physical_from_sine(coef: np.ndarray, ops: _Operators) -> np.ndarray:
    """z-derivative of a sine-represented field, in full physical (Nz, Nx) space."""
    z_deriv_phys_mixed = ops.C @ (ops.m_pi[:, None] * coef)  # (Nz, Nkx)
    return irfft(z_deriv_phys_mixed, axis=-1, workers=ops.fft_workers)


def dealias(coef: np.ndarray, ops: _Operators) -> np.ndarray:
    return coef * ops.dealias_mask


def rhs_nonlinear(omega_hat: np.ndarray, theta_hat: np.ndarray, ops: _Operators):
    """PURELY nonlinear advection terms `N_omega = -u.grad(omega)`,
    `N_theta = -u.grad(theta)` (pseudospectral, dealiased). Buoyancy
    (`Pr*Ra*d(theta)/dx`) and the background-gradient advection term
    (`+w` in the theta equation) are BOTH linear in `(omega_hat,
    theta_hat)` and are handled by `_implicit_coupling_matrices`/
    `cnab2_step` instead -- see that function's docstring for why: at
    this project's target `Ra=3e4`, the fastest LINEAR growth rate across
    all resolved modes is ~93 (checked directly), which would force
    `dt << 1/93` for stability if treated explicitly (confirmed
    empirically: the target Ra/Pr blew up within ~30 steps at
    `dt=cfg.dt_max=5e-3` when buoyancy was explicit). Treating the linear
    coupling implicitly (Crank-Nicolson, unconditionally A-stable) removes
    that constraint entirely; only the genuinely nonlinear term -- whose
    stability constraint IS the ordinary advective CFL this module's
    `cfl_dt` already tracks -- remains explicit.

    NOTE: `u_phys`/`w_phys` here are computed from the DEALIASED `psi_d`,
    deliberately NOT shared with `integrate`'s own (non-dealiased)
    `velocities_physical` call used for the CFL estimate, even though
    both derive from the same `omega_hat` -- reusing the non-dealiased
    version here would silently reintroduce aliasing error into the
    nonlinear product (2/3-rule dealiasing requires EVERY factor entering
    a quadratic product to be truncated, not just some of them). Costs a
    second small transform pair per step; correctness was judged worth
    more than that increment at this project's target resolution."""
    psi_hat = omega_hat / ops.neg_lap  # laplacian(psi) = -omega -> psi_hat = omega_hat / neg_lap

    omega_d = dealias(omega_hat, ops)
    theta_d = dealias(theta_hat, ops)
    psi_d = dealias(psi_hat, ops)

    u_phys = dz_physical_from_sine(psi_d, ops)  # u = d(psi)/dz
    w_hat_sine = -1j * ops.kx[None, :] * psi_d  # w = -d(psi)/dx, stays sine-in-z
    w_phys = irfft(ops.S @ w_hat_sine, axis=-1, workers=ops.fft_workers)

    domega_dx_phys = irfft(ops.S @ (1j * ops.kx[None, :] * omega_d), axis=-1, workers=ops.fft_workers)
    domega_dz_phys = dz_physical_from_sine(omega_d, ops)
    dtheta_dx_phys = irfft(ops.S @ (1j * ops.kx[None, :] * theta_d), axis=-1, workers=ops.fft_workers)
    dtheta_dz_phys = dz_physical_from_sine(theta_d, ops)

    adv_omega_phys = u_phys * domega_dx_phys + w_phys * domega_dz_phys
    adv_theta_phys = u_phys * dtheta_dx_phys + w_phys * dtheta_dz_phys

    N_omega = -sine_from_physical(adv_omega_phys, ops)
    N_theta = -sine_from_physical(adv_theta_phys, ops)

    return N_omega, N_theta


def _implicit_coupling_matrices(dt: float, ops: _Operators, Ra: float, Pr: float):
    """Per-mode 2x2 linear system `L[m,kx] = [[-Pr*lap, Pr*Ra*i*kx],
    [-i*kx/lap, -lap]]` (diffusion + buoyancy + background-gradient
    advection, all exactly linear in `(omega_hat, theta_hat)` -- the same
    matrix `tests/unit/test_rayleigh_benard.py::_linear_system_matrix`
    validates against the textbook critical Ra), Crank-Nicolson-
    discretized: returns `(Mm_inv_a, Mm_inv_b, Mm_inv_c, Mm_inv_d,
    Mp_a, Mp_b, Mp_c, Mp_d)`, the elementwise entries of
    `(I - dt/2*L)^-1` and `(I + dt/2*L)` respectively (each shape
    `(Nz, Nkx)`), via the closed-form 2x2 inverse -- vectorized across
    every mode at once, no per-mode loop."""
    lap = ops.neg_lap
    a = -Pr * lap
    b = Pr * Ra * 1j * ops.kx[None, :] * np.ones_like(lap)
    c = -1j * ops.kx[None, :] * np.ones_like(lap) / lap
    d = -lap

    h = 0.5 * dt
    Mp_a, Mp_b, Mp_c, Mp_d = 1 + h * a, h * b, h * c, 1 + h * d
    Mm_a, Mm_b, Mm_c, Mm_d = 1 - h * a, -h * b, -h * c, 1 - h * d
    det = Mm_a * Mm_d - Mm_b * Mm_c
    Mm_inv_a, Mm_inv_b, Mm_inv_c, Mm_inv_d = Mm_d / det, -Mm_b / det, -Mm_c / det, Mm_a / det

    return (Mm_inv_a, Mm_inv_b, Mm_inv_c, Mm_inv_d, Mp_a, Mp_b, Mp_c, Mp_d)


def cnab2_step(omega_hat, theta_hat, N_omega_prev, N_theta_prev, dt: float, ops: _Operators, Ra: float, Pr: float):
    """One CNAB2 step: Crank-Nicolson on the exact linear (diffusion +
    buoyancy + background-advection) coupling, 2nd-order Adams-Bashforth
    on the nonlinear advection term. `N_*_prev` is `None` on the very
    first step (falls back to a first-order explicit-trapezoid bootstrap
    for that step only)."""
    N_omega, N_theta = rhs_nonlinear(omega_hat, theta_hat, ops)

    if N_omega_prev is None:
        ab_omega, ab_theta = N_omega, N_theta  # bootstrap: dt/2*(3N-N) with N_prev=N -> dt*N
    else:
        ab_omega = 1.5 * N_omega - 0.5 * N_omega_prev
        ab_theta = 1.5 * N_theta - 0.5 * N_theta_prev

    Mm_inv_a, Mm_inv_b, Mm_inv_c, Mm_inv_d, Mp_a, Mp_b, Mp_c, Mp_d = _implicit_coupling_matrices(dt, ops, Ra, Pr)

    rhs_omega = Mp_a * omega_hat + Mp_b * theta_hat + dt * ab_omega
    rhs_theta = Mp_c * omega_hat + Mp_d * theta_hat + dt * ab_theta

    omega_next = Mm_inv_a * rhs_omega + Mm_inv_b * rhs_theta
    theta_next = Mm_inv_c * rhs_omega + Mm_inv_d * rhs_theta

    return omega_next, theta_next, N_omega, N_theta


def velocities_physical(omega_hat: np.ndarray, ops: _Operators) -> tuple[np.ndarray, np.ndarray]:
    """u, w in full physical (Nz, Nx) space, from omega's sine coefficients."""
    psi_hat = omega_hat / ops.neg_lap
    u_phys = dz_physical_from_sine(psi_hat, ops)
    w_hat_sine = -1j * ops.kx[None, :] * psi_hat
    w_phys = irfft(ops.S @ w_hat_sine, axis=-1, workers=ops.fft_workers)
    return u_phys, w_phys


def cfl_dt(u_phys: np.ndarray, w_phys: np.ndarray, cfg: RayleighBenardConfig) -> float:
    dx = cfg.Lx / cfg.Nx
    dz = cfg.Lz / (cfg.Nz + 1)
    max_u = float(np.max(np.abs(u_phys)))
    max_w = float(np.max(np.abs(w_phys)))
    denom = max_u / dx + max_w / dz + 1e-12
    dt = cfg.cfl_target / denom
    return float(np.clip(dt, cfg.dt_min, cfg.dt_max))


def default_initial_condition(cfg: RayleighBenardConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Small-amplitude random noise in every retained sine/rfft mode
    (dealiased range only), returned as `(omega_hat, theta_hat)`. The
    conductive base state (theta=0 everywhere in this perturbation
    variable, omega=0, no flow) is exactly a fixed point of the full
    nonlinear equations, so ANY nonzero perturbation is what triggers
    convection -- matches the brief's "small-amplitude random noise
    superimposed on the conductive profile" request directly."""
    ops = get_operators(cfg)
    shape = ops.neg_lap.shape
    real = rng.normal(scale=cfg.perturbation_amplitude, size=shape)
    imag = rng.normal(scale=cfg.perturbation_amplitude, size=shape)
    theta_hat = dealias((real + 1j * imag).astype(np.complex128), ops)
    real2 = rng.normal(scale=cfg.perturbation_amplitude, size=shape)
    imag2 = rng.normal(scale=cfg.perturbation_amplitude, size=shape)
    omega_hat = dealias((real2 + 1j * imag2).astype(np.complex128), ops)
    return omega_hat, theta_hat


def integrate(
    omega_hat0: np.ndarray,
    theta_hat0: np.ndarray,
    cfg: RayleighBenardConfig,
    total_time: float,
    *,
    n_snapshots: int | None = None,
    max_abs_velocity: float = 1e3,
) -> dict:
    """Integrate from `t=0` to `t=total_time`, recording a snapshot every
    `cfg.dt_snap` physical time units (the last sub-step before each
    snapshot boundary is shrunk to land on it exactly). Returns a dict
    with `t` (n_snap,), `omega_hat`/`theta_hat` (n_snap, Nz, Nkx) complex,
    and `dt_history` (list of every sub-step dt actually taken, for
    diagnostics). Raises `RuntimeError` on non-finite or blown-up state
    (ground rule 2: fail loudly, no silent nan swallowing)."""
    ops = get_operators(cfg)
    if n_snapshots is None:
        n_snapshots = int(round(total_time / cfg.dt_snap))
    snap_times = np.arange(1, n_snapshots + 1) * cfg.dt_snap

    omega_hat, theta_hat = omega_hat0.copy(), theta_hat0.copy()
    N_omega_prev, N_theta_prev = None, None
    t = 0.0
    dt = cfg.dt0
    dt_prev_taken: float | None = None
    dt_history = []

    omega_out = np.empty((n_snapshots, *omega_hat.shape), dtype=np.complex128)
    theta_out = np.empty((n_snapshots, *theta_hat.shape), dtype=np.complex128)

    step_count = 0
    for snap_idx, t_target in enumerate(snap_times):
        while t < t_target - 1e-13:
            if step_count % cfg.cfl_check_every == 0:
                u_phys, w_phys = velocities_physical(omega_hat, ops)
                dt = cfl_dt(u_phys, w_phys, cfg)
            dt_this = min(dt, t_target - t)
            # Reset the AB2 bootstrap whenever the step size changes
            # materially from the last step actually taken (found
            # 2026-09-23: clamping dt_this to land exactly on a snapshot
            # boundary creates a step-size discontinuity the FIXED-
            # coefficient AB2 extrapolation `1.5*N^n - 0.5*N^{n-1}` isn't
            # valid across -- it implicitly assumes uniform step size.
            # Confirmed directly: at this project's target Ra=3e4, a
            # single far-off snapshot target integrates cleanly to a
            # bounded saturated state, while frequent snapshot boundaries
            # (the same physical config, only the snapshot cadence
            # differs) reliably blow up -- the mismatched extrapolation
            # injects error right during the most sensitive part of the
            # transient growth phase, and it compounds. Costs at most one
            # extra first-order-accurate step per boundary crossing.)
            if dt_prev_taken is not None and not (0.8 <= dt_this / dt_prev_taken <= 1.25):
                N_omega_prev, N_theta_prev = None, None
            omega_hat, theta_hat, N_omega_prev, N_theta_prev = cnab2_step(
                omega_hat, theta_hat, N_omega_prev, N_theta_prev, dt_this, ops, cfg.Ra, cfg.Pr
            )
            dt_prev_taken = dt_this
            if not (np.isfinite(omega_hat).all() and np.isfinite(theta_hat).all()):
                raise RuntimeError(
                    f"Rayleigh-Benard integration produced non-finite values at t={t + dt_this:.4f}."
                )
            # Boundedness check on the PHYSICAL velocity field, not the
            # raw spectral coefficients (found 2026-09-23: an earlier
            # version checked max(|omega_hat|, |theta_hat|) directly --
            # those are transform-normalization-dependent numbers with no
            # fixed physical scale, and a run that was genuinely bounded
            # and saturated in physical space (omega_phys ~500-550, a
            # healthy statistically-steady state at this Ra) had its raw
            # spectral coefficient climb past 1e5 anyway, triggering a
            # false-positive "blowup". Velocity is already computed for
            # the CFL estimate above whenever step_count just hit a check
            # boundary; recomputed fresh here otherwise so this check
            # never runs on stale data (cheap either way at this
            # resolution).
            if step_count % cfg.cfl_check_every != 0:
                u_phys, w_phys = velocities_physical(omega_hat, ops)
            max_abs_vel = max(float(np.max(np.abs(u_phys))), float(np.max(np.abs(w_phys))))
            if not np.isfinite(max_abs_vel):
                raise RuntimeError(
                    f"Rayleigh-Benard integration produced non-finite velocity at t={t + dt_this:.4f}."
                )
            if max_abs_vel > max_abs_velocity:
                raise RuntimeError(
                    f"Rayleigh-Benard integration exceeded max_abs_velocity={max_abs_velocity:.3g} "
                    f"(reached {max_abs_vel:.3g}) at t={t + dt_this:.4f} -- the solution is not on a "
                    "bounded attractor; check Ra/Pr/resolution/dt before trusting this run."
                )
            t += dt_this
            dt_history.append(dt_this)
            step_count += 1
        omega_out[snap_idx] = omega_hat
        theta_out[snap_idx] = theta_hat

    return {
        "t": snap_times,
        "omega_hat": omega_out,
        "theta_hat": theta_out,
        "dt_history": np.array(dt_history),
    }


def spinup(cfg: RayleighBenardConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Integrate from a small random perturbation for `cfg.spinup_time`
    and return the final `(omega_hat, theta_hat)`, discarding the
    transient -- same role as KS/L96's own `spinup`."""
    omega_hat0, theta_hat0 = default_initial_condition(cfg, rng)
    result = integrate(omega_hat0, theta_hat0, cfg, cfg.spinup_time, n_snapshots=1)
    return result["omega_hat"][-1], result["theta_hat"][-1]


def to_physical_fields(omega_hat: np.ndarray, theta_hat: np.ndarray, ops: _Operators) -> dict:
    """Decode sine/rfft coefficients to physical `(Nz, Nx)` fields:
    `omega`, `theta`, `u`, `w`, `psi`. For visualization/diagnostics/
    dataset export -- NOT dealiased (full resolution)."""
    psi_hat = omega_hat / ops.neg_lap
    omega_phys = irfft(ops.S @ omega_hat, axis=-1, workers=ops.fft_workers)
    theta_phys = irfft(ops.S @ theta_hat, axis=-1, workers=ops.fft_workers)
    psi_phys = irfft(ops.S @ psi_hat, axis=-1, workers=ops.fft_workers)
    u_phys, w_phys = velocities_physical(omega_hat, ops)
    return {"omega": omega_phys, "theta": theta_phys, "psi": psi_phys, "u": u_phys, "w": w_phys}
