"""Validation tests for the 2D Rayleigh-Benard solver (ground rule 1: no
untested numerics). The critical-Rayleigh-number linear-stability test is
the primary correctness gate, directly analogous to KS's own
`test_L100_kaplan_yorke` -- see `RayleighBenardConfig`'s docstring for the
exact textbook target (`Ra_c = 27*pi**4/4`) and why this solver's specific
choice of boundary conditions/aspect ratio makes that value exactly
reachable.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import irfft, rfft

from ks_latent.config import RayleighBenardConfig
from ks_latent.solver.rayleigh_benard import (
    _sine_cosine_matrices,
    cnab2_step,
    default_initial_condition,
    get_operators,
    integrate,
    physical_from_sine,
    sine_from_physical,
    spinup,
    to_physical_fields,
)


def test_sine_cosine_matrices_orthonormal_and_derivative():
    """S must be exactly orthonormal (S.T @ S = I, a standard DST-I
    identity); C must synthesize the analytic z-derivative of a single
    sine mode to near machine precision."""
    Nz = 32
    S, C, m_pi, z_grid = _sine_cosine_matrices(Nz)
    assert np.allclose(S.T @ S, np.eye(Nz), atol=1e-12)

    # f(z) = sin(3*pi*z) -> coefficients = a pure m=3 spike (S's columns
    # carry a sqrt(2/(Nz+1)) normalization, so the spike's magnitude is
    # sqrt((Nz+1)/2), not 1 -- check the SHAPE (only index m-1 nonzero)
    # rather than hand-deriving that prefactor).
    m = 3
    f = np.sin(m * np.pi * z_grid)
    coeffs = S.T @ f
    off_target = np.delete(coeffs, m - 1)
    assert np.allclose(off_target, 0.0, atol=1e-10)
    assert abs(coeffs[m - 1]) > 1.0

    # d/dz[sin(3*pi*z)] = 3*pi*cos(3*pi*z), synthesized via C.
    df_dz_numeric = C @ (m_pi * coeffs)
    df_dz_exact = m * np.pi * np.cos(m * np.pi * z_grid)
    assert np.allclose(df_dz_numeric, df_dz_exact, atol=1e-10)


def test_sine_rfft_roundtrip():
    """physical_from_sine(sine_from_physical(x)) == x to near machine
    precision, for a random real physical field."""
    cfg = RayleighBenardConfig(Nx=16, Nz=12)
    ops = get_operators(cfg)
    rng = np.random.default_rng(0)
    phys = rng.normal(size=(cfg.Nz, cfg.Nx))
    coef = sine_from_physical(phys, ops)
    recovered = physical_from_sine(coef, ops)
    assert np.allclose(phys, recovered, atol=1e-10)


def test_zero_state_is_a_fixed_point():
    """The purely conductive base state (omega=theta=0 everywhere) must
    be an exact fixed point of the full nonlinear equations -- zero stays
    zero under cnab2_step, for any Ra/Pr."""
    cfg = RayleighBenardConfig(Nx=16, Nz=12, Ra=1e5, Pr=0.7)
    ops = get_operators(cfg)
    zero = np.zeros(ops.neg_lap.shape, dtype=np.complex128)
    omega_next, theta_next, _, _ = cnab2_step(zero, zero, None, None, 1e-3, ops, cfg.Ra, cfg.Pr)
    assert np.allclose(omega_next, 0.0, atol=1e-14)
    assert np.allclose(theta_next, 0.0, atol=1e-14)


def _linear_system_matrix(k: float, m: int, Ra: float, Pr: float) -> np.ndarray:
    lap = k**2 + (m * np.pi) ** 2
    return np.array(
        [
            [-Pr * lap, Pr * Ra * 1j * k],
            [-1j * k / lap, -lap],
        ]
    )


def _linear_growth_rate(k: float, m: int, Ra: float, Pr: float) -> float:
    """Largest real part of the eigenvalues of the linearized 2x2
    (omega_hat, theta_hat) system at a single (k, m) mode -- derived
    directly from the governing equations in RayleighBenardConfig's
    docstring (see this module's own header comment / the section script
    that built it for the full by-hand derivation). Independent of the
    actual solver code: used as the analytic target it must reproduce."""
    eigs = np.linalg.eigvals(_linear_system_matrix(k, m, Ra, Pr))
    return float(np.max(eigs.real))


def _dominant_eigenvector(k: float, m: int, Ra: float, Pr: float) -> np.ndarray:
    """(omega_hat, theta_hat) eigenvector of the DOMINANT (largest real
    part) eigenvalue -- this system has two distinct real eigenvalues at
    a generic Ra (verified numerically: e.g. -4.34 and -25.27 at
    Ra=0.5*Ra_c, Pr=1), so a generic initial condition is a MIX of both
    modes and its early-time decay rate is contaminated by the faster
    one; only initializing exactly along this eigenvector gives pure
    single-rate exponential behavior from t=0."""
    eigvals, eigvecs = np.linalg.eig(_linear_system_matrix(k, m, Ra, Pr))
    idx = int(np.argmax(eigvals.real))
    return eigvecs[:, idx]


def test_critical_rayleigh_number_matches_textbook_value():
    """The analytic 2x2 linear-stability reduction (independent of the
    numerical solver) must give marginal stability (growth rate ~= 0) at
    the textbook free-free critical Rayleigh number Ra_c = 27*pi**4/4 at
    the critical wavenumber k_c = pi/sqrt(2), m=1 -- Chandrasekhar (1961),
    ch. II. This is the external, known-answer check: not derived from
    this codebase, a citable closed-form textbook result."""
    k_c = np.pi / np.sqrt(2.0)
    Ra_c = 27.0 * np.pi**4 / 4.0
    sigma = _linear_growth_rate(k_c, 1, Ra_c, Pr=1.0)  # growth rate at marginal stability is Pr-independent
    assert abs(sigma) < 1e-8

    sigma_below = _linear_growth_rate(k_c, 1, 0.9 * Ra_c, Pr=1.0)
    sigma_above = _linear_growth_rate(k_c, 1, 1.1 * Ra_c, Pr=1.0)
    assert sigma_below < -1e-3
    assert sigma_above > 1e-3


@pytest.mark.slow
def test_solver_reproduces_linear_growth_rate_below_and_above_critical():
    """The REAL nonlinear pseudospectral solver, given a tiny-amplitude
    single-mode initial perturbation (nonlinear terms ~ amplitude^2,
    negligible), must reproduce the analytic linear growth rate at the
    critical wavenumber for both a sub- and super-critical Ra. This
    validates the actual production code path (not a separate linearized
    implementation) against the textbook-anchored target above."""
    for ra_frac, expect_sign in [(0.5, -1), (3.0, +1)]:
        cfg = RayleighBenardConfig(
            Ra=ra_frac * 27.0 * np.pi**4 / 4.0, Pr=1.0, Nx=8, Nz=16,
            dt0=2e-4, dt_max=2e-4, cfl_check_every=10_000_000,
            snapshot_dt=0.02, perturbation_amplitude=1e-6,
        )
        ops = get_operators(cfg)
        k_c = 2 * np.pi / cfg.Lx  # fundamental x-wavenumber = k_c by this config's aspect ratio
        eigvec = _dominant_eigenvector(k_c, 1, cfg.Ra, cfg.Pr)
        # Single-mode initial condition, exactly along the dominant
        # eigenvector (see _dominant_eigenvector's docstring for why a
        # naive (omega=0, theta=eps) IC does NOT give pure single-rate
        # decay here): kx index 1 (fundamental = k_c), z-mode m=1.
        omega_hat0 = np.zeros(ops.neg_lap.shape, dtype=np.complex128)
        theta_hat0 = np.zeros(ops.neg_lap.shape, dtype=np.complex128)
        omega_hat0[0, 1] = cfg.perturbation_amplitude * eigvec[0]
        theta_hat0[0, 1] = cfg.perturbation_amplitude * eigvec[1]

        result = integrate(omega_hat0, theta_hat0, cfg, total_time=0.3, max_abs_velocity=1e3)
        amp = np.abs(result["theta_hat"][:, 0, 1])
        assert np.all(amp > 0)
        log_amp = np.log(amp)
        # Linear regression of log-amplitude vs time -> measured growth rate.
        slope = np.polyfit(result["t"], log_amp, 1)[0]
        analytic = _linear_growth_rate(k_c, 1, cfg.Ra, cfg.Pr)
        assert np.sign(slope) == expect_sign
        assert slope == pytest.approx(analytic, rel=0.03, abs=0.05)


def test_spinup_reaches_bounded_nontrivial_state():
    cfg = RayleighBenardConfig(Nx=16, Nz=12, spinup_time=0.5)
    rng = np.random.default_rng(1)
    omega_hat, theta_hat = spinup(cfg, rng)
    assert np.isfinite(omega_hat).all()
    assert np.isfinite(theta_hat).all()
    assert np.abs(theta_hat).max() > 0.0


def test_to_physical_fields_shapes_and_finite():
    cfg = RayleighBenardConfig(Nx=16, Nz=12, spinup_time=0.3)
    rng = np.random.default_rng(2)
    omega_hat, theta_hat = spinup(cfg, rng)
    ops = get_operators(cfg)
    fields = to_physical_fields(omega_hat, theta_hat, ops)
    for name in ["omega", "theta", "psi", "u", "w"]:
        assert fields[name].shape == (cfg.Nz, cfg.Nx)
        assert np.isfinite(fields[name]).all()


def test_integrate_snapshot_count_and_time_grid():
    cfg = RayleighBenardConfig(Nx=16, Nz=12, snapshot_dt=0.05)
    rng = np.random.default_rng(3)
    omega_hat0, theta_hat0 = default_initial_condition(cfg, rng)
    result = integrate(omega_hat0, theta_hat0, cfg, total_time=0.2)
    assert result["omega_hat"].shape[0] == 4
    assert np.allclose(result["t"], [0.05, 0.10, 0.15, 0.20], atol=1e-9)
