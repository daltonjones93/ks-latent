"""Solver tests (brief §3.2). Fast tests only; slow Lyapunov/D_KY replication
targets live in tests/replication/test_gate1_kaplan_yorke.py (brief §18) even
though the brief's §3.2 table lists them alongside these -- both are
@pytest.mark.slow and neither runs in `make test-fast`.
"""

from __future__ import annotations

import mpmath
import numpy as np
import pytest
from scipy.optimize import brentq

from ks_latent.config import KSConfig
from ks_latent.solver.ks import (
    default_initial_condition,
    etdrk4_coefficients,
    get_operator,
    integrate,
    linear_operator,
    wavenumbers,
)


def test_linear_operator_spectrum():
    """Most unstable continuous wavenumber is k = 1/sqrt(2) (brief table)."""
    # Lhat(k) = k**2 - k**4; dLhat/dk = 2k - 4k**3 = 2k(1 - 2k**2) = 0 at k=1/sqrt(2).
    root = brentq(lambda k: 2 * k - 4 * k**3, 0.1, 2.0, xtol=1e-14, rtol=1e-14)
    assert root == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-10)
    wavelength = 2 * np.pi / root
    assert wavelength == pytest.approx(2 * np.sqrt(2) * np.pi, abs=1e-9)


def _mpmath_coeffs(z: complex, h: float, dps: int = 50):
    mpmath.mp.dps = dps
    zz = mpmath.mpc(z)
    exp = mpmath.exp
    if abs(zz) < 1e-30:
        # Exact Taylor limits at z=0 (standard ETDRK4 removable-singularity values).
        return h * 0.5, h / 6.0, h / 6.0, h / 6.0
    Q = h * (exp(zz / 2) - 1) / zz
    f1 = h * (-4 - zz + exp(zz) * (4 - 3 * zz + zz**2)) / zz**3
    f2 = h * (2 + zz + exp(zz) * (-2 + zz)) / zz**3
    f3 = h * (-4 - 3 * zz - zz**2 + exp(zz) * (4 - zz)) / zz**3
    return complex(Q).real, complex(f1).real, complex(f2).real, complex(f3).real


def test_etdrk4_coefficients_no_cancellation():
    """Coefficients at Lhat ~ 0 match an mpmath high-precision reference to 1e-12."""
    h = 0.05
    for lhat_val in [0.0, 1e-4, -1e-4, 1e-3]:
        coeffs = etdrk4_coefficients(np.array([lhat_val]), h, M=32)
        z = h * lhat_val
        Q_ref, f1_ref, f2_ref, f3_ref = _mpmath_coeffs(z, h)
        assert coeffs.Q[0] == pytest.approx(Q_ref, abs=1e-12)
        assert coeffs.f1[0] == pytest.approx(f1_ref, abs=1e-12)
        assert coeffs.f2[0] == pytest.approx(f2_ref, abs=1e-12)
        assert coeffs.f3[0] == pytest.approx(f3_ref, abs=1e-12)


def test_mean_conservation():
    cfg = KSConfig(L=100.0, NX=256, dt=0.05, snapshot_every=20)
    rng = np.random.default_rng(0)
    u0 = default_initial_condition(cfg, rng)
    mean0 = u0.mean()
    traj = integrate(u0, cfg, n_steps=10_000, snapshot_every=500)
    means = traj.mean(axis=1)
    assert np.max(np.abs(means - mean0)) < 1e-8


def test_translation_equivariance():
    cfg = KSConfig(L=100.0, NX=256, dt=0.05, snapshot_every=5)
    rng = np.random.default_rng(1)
    u0 = default_initial_condition(cfg, rng)
    shift = 37
    traj_a = integrate(np.roll(u0, shift), cfg, n_steps=200)
    traj_b = integrate(u0, cfg, n_steps=200)
    assert np.allclose(traj_a, np.roll(traj_b, shift, axis=1), atol=1e-10)


def test_convergence_in_dt():
    """Observed order >= 3.5 vs. a dt/8 reference, short non-chaotic window.

    Uses a smooth, moderate-amplitude two-mode IC on L=22 rather than the
    default random low-mode IC: a decaying, near-linear regime makes the
    nonlinear (u**2) forcing negligible relative to floating-point noise
    at fine dt, which corrupts the *measured* order without indicating any
    solver bug -- confirmed separately against an independent scipy
    `solve_ivp(DOP853, rtol=1e-13)` reference on this same IC, where order
    approaches 4 cleanly as dt -> 0.
    """
    L, NX = 22.0, 32
    x = np.arange(NX) * L / NX
    u0 = 0.5 * np.cos(2 * np.pi * x / L) + 0.2 * np.sin(4 * np.pi * x / L)
    t_final = 1.0
    base_cfg = KSConfig(L=L, NX=NX, dt=0.1, snapshot_every=1)

    ref_cfg = base_cfg.replace(dt=base_cfg.dt / 8)
    n_ref = round(t_final / ref_cfg.dt)
    ref = integrate(u0, ref_cfg, n_steps=n_ref, snapshot_every=n_ref)[-1]

    errors = []
    for dt in [base_cfg.dt, base_cfg.dt / 2]:
        cfg = base_cfg.replace(dt=dt)
        n_steps = round(t_final / dt)
        out = integrate(u0, cfg, n_steps=n_steps, snapshot_every=n_steps)[-1]
        errors.append(np.linalg.norm(out - ref))

    order = np.log2(errors[0] / errors[1])
    assert order >= 3.5, f"observed order {order:.2f} < 3.5 ({errors=})"


def test_small_L_decay():
    """For L < 2*pi, the zero solution is linearly stable and u -> 0."""
    cfg = KSConfig(L=2.0, NX=32, dt=0.01, snapshot_every=50)
    rng = np.random.default_rng(3)
    u0 = default_initial_condition(cfg, rng)
    traj = integrate(u0, cfg, n_steps=5000, snapshot_every=500)
    assert np.linalg.norm(traj[-1]) < 1e-6 * max(np.linalg.norm(traj[0]), 1e-12) + 1e-10


def test_energy_spectrum_peak():
    cfg = KSConfig(L=100.0, NX=512, dt=0.05, snapshot_every=5)
    rng = np.random.default_rng(4)
    from ks_latent.solver.ks import spinup

    u0 = spinup(cfg.replace(spinup_time=200.0), rng)
    traj = integrate(u0, cfg, n_steps=4000, snapshot_every=5)
    spec = np.mean(np.abs(np.fft.fft(traj, axis=1)) ** 2, axis=0)
    k = wavenumbers(cfg.L, cfg.NX)
    pos = k > 0
    k_peak = k[pos][np.argmax(spec[pos])]
    assert k_peak == pytest.approx(1.0 / np.sqrt(2.0), abs=0.15)


@pytest.mark.unit
def test_operator_cache_is_keyed_by_L_NX_dt():
    cfg_a = KSConfig(L=100.0, NX=64, dt=0.05)
    cfg_b = KSConfig(L=100.0, NX=64, dt=0.05)
    cfg_c = KSConfig(L=100.0, NX=64, dt=0.025)
    op_a = get_operator(cfg_a)
    op_b = get_operator(cfg_b)
    op_c = get_operator(cfg_c)
    assert op_a is op_b
    assert op_a is not op_c
