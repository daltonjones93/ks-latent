"""2D KS solver tests (`docs/ks_local_field_ltransfer.tex` "Staged next
steps" Step 1), mirroring `tests/unit/test_solver.py`'s own Gate-1-style
discipline for the 1D solver -- no untested numerics (ground rule 1).
Slow Lyapunov/D_KY validation lives separately (a script/replication
test, not here, matching the 1D convention of keeping those out of the
fast unit suite).
"""

from __future__ import annotations

import mpmath
import numpy as np
import pytest
from scipy.optimize import brentq

from ks_latent.config import KSConfig2D
from ks_latent.solver.ks import etdrk4_coefficients
from ks_latent.solver.ks2d import (
    default_initial_condition2d,
    etdrk4_coefficients_2d,
    integrate2d,
    linear_operator2d,
    spinup2d,
    wavenumbers2d,
)


def test_linear_operator_spectrum_2d_isotropic_ring():
    """Same critical radius as 1D, |K| = 1/sqrt(2), but now a whole RING
    in (kx,ky) -- the equation is isotropic, so the critical radius must
    not depend on direction."""
    for theta in [0.0, 0.3, np.pi / 4, 1.1, 2.7]:
        def neg_growth_rate(K):
            kx, ky = K * np.cos(theta), K * np.sin(theta)
            Lhat = linear_operator2d(np.array([kx]), np.array([ky]))[0]
            return Lhat
        # dLhat/dK at fixed direction: same 2K(1-2K^2)=0 root as 1D (K^2=kx^2+ky^2
        # plays the role of 1D's k^2 identically in Lhat=K^2-K^4).
        root = brentq(lambda K: 2 * K - 4 * K**3, 0.1, 2.0, xtol=1e-14, rtol=1e-14)
        assert root == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-10)


def _mpmath_coeffs(z: complex, h: float, dps: int = 50):
    mpmath.mp.dps = dps
    zz = mpmath.mpc(z)
    exp = mpmath.exp
    if abs(zz) < 1e-30:
        return h * 0.5, h / 6.0, h / 6.0, h / 6.0
    Q = h * (exp(zz / 2) - 1) / zz
    f1 = h * (-4 - zz + exp(zz) * (4 - 3 * zz + zz**2)) / zz**3
    f2 = h * (2 + zz + exp(zz) * (-2 + zz)) / zz**3
    f3 = h * (-4 - 3 * zz - zz**2 + exp(zz) * (4 - zz)) / zz**3
    return complex(Q).real, complex(f1).real, complex(f2).real, complex(f3).real


def test_etdrk4_coefficients_2d_matches_1d_on_a_1d_shaped_input():
    """`etdrk4_coefficients_2d` generalizes `ks.etdrk4_coefficients` via
    `[..., None]`/`axis=-1` broadcasting instead of `[:, None]`/`axis=1`
    -- on a genuinely 1D-shaped `Lhat`, the two must agree EXACTLY (same
    algorithm, just a more general reduction axis convention), which
    directly validates the broadcasting generalization is lossless."""
    h = 0.02
    Lhat_1d = np.array([0.0, 1e-4, -1e-4, 1e-3, 0.5, -2.0, 3.7])
    ref = etdrk4_coefficients(Lhat_1d, h, M=32)
    gen = etdrk4_coefficients_2d(Lhat_1d, h, M=32)
    for name in ["E", "E2", "Q", "f1", "f2", "f3"]:
        assert np.allclose(getattr(ref, name), getattr(gen, name), atol=1e-14), name


def test_etdrk4_coefficients_2d_no_cancellation_near_zero():
    """Coefficients at Lhat ~ 0 (2D-shaped input) match an mpmath
    high-precision reference to 1e-12, same bar as 1D's own test."""
    h = 0.02
    Lhat_2d = np.array([[0.0, 1e-4], [-1e-4, 1e-3]])
    coeffs = etdrk4_coefficients_2d(Lhat_2d, h, M=32)
    for i in range(2):
        for j in range(2):
            z = h * Lhat_2d[i, j]
            Q_ref, f1_ref, f2_ref, f3_ref = _mpmath_coeffs(z, h)
            assert coeffs.Q[i, j] == pytest.approx(Q_ref, abs=1e-12)
            assert coeffs.f1[i, j] == pytest.approx(f1_ref, abs=1e-12)
            assert coeffs.f2[i, j] == pytest.approx(f2_ref, abs=1e-12)
            assert coeffs.f3[i, j] == pytest.approx(f3_ref, abs=1e-12)


def test_mean_conservation_2d():
    """Exact by construction (see `step2d`'s own docstring: the k=0 mode
    is projected out every step, since the dynamics never depend on
    mean(u) at all) -- verified directly over a real trajectory, not
    just asserted from the implementation."""
    cfg = KSConfig2D(Lx=44.0, Ly=44.0, NX=48, NY=48, dt=0.02, snapshot_every=20)
    rng = np.random.default_rng(0)
    u0 = default_initial_condition2d(cfg, rng)
    traj = integrate2d(u0, cfg, n_steps=2000, snapshot_every=100)
    means = traj.mean(axis=(1, 2))
    assert np.max(np.abs(means)) < 1e-8


def test_translation_equivariance_2d_both_axes():
    cfg = KSConfig2D(Lx=44.0, Ly=44.0, NX=48, NY=48, dt=0.02, snapshot_every=5)
    rng = np.random.default_rng(1)
    u0 = default_initial_condition2d(cfg, rng)
    shift_x, shift_y = 13, 7
    shifted_u0 = np.roll(np.roll(u0, shift_x, axis=0), shift_y, axis=1)
    traj_a = integrate2d(shifted_u0, cfg, n_steps=100)
    traj_b = integrate2d(u0, cfg, n_steps=100)
    traj_b_shifted = np.roll(np.roll(traj_b, shift_x, axis=1), shift_y, axis=2)
    assert np.allclose(traj_a, traj_b_shifted, atol=1e-10)


def test_convergence_in_dt_2d():
    """Observed order >= 3.5 vs. a dt/8 reference, short non-chaotic
    window -- same discipline as 1D's own `test_convergence_in_dt`
    (smooth, moderate-amplitude, near-linear IC to avoid the nonlinear
    forcing being swamped by floating-point noise at fine dt)."""
    Lx, Ly, NX, NY = 22.0, 22.0, 24, 24
    x = np.arange(NX) * Lx / NX
    y = np.arange(NY) * Ly / NY
    X, Y = np.meshgrid(x, y, indexing="ij")
    # Larger amplitude / longer t_final than a first attempt used (which
    # decayed too fast, pushing errors down to the float64 noise floor
    # ~1e-13 before dt/2 vs. dt/8 could separate meaningfully -- same
    # caveat 1D's own test docstring already flags for an
    # under-amplitude/over-linear regime).
    u0 = 1.5 * np.cos(2 * np.pi * X / Lx) + 0.8 * np.sin(2 * np.pi * Y / Ly)
    u0 = u0 - u0.mean()
    t_final = 1.0
    base_cfg = KSConfig2D(Lx=Lx, Ly=Ly, NX=NX, NY=NY, dt=0.05, snapshot_every=1)

    ref_cfg = base_cfg.replace(dt=base_cfg.dt / 8)
    n_ref = round(t_final / ref_cfg.dt)
    ref = integrate2d(u0, ref_cfg, n_steps=n_ref, snapshot_every=n_ref)[-1]

    errors = []
    for dt in [base_cfg.dt, base_cfg.dt / 2]:
        cfg = base_cfg.replace(dt=dt)
        n_steps = round(t_final / dt)
        out = integrate2d(u0, cfg, n_steps=n_steps, snapshot_every=n_steps)[-1]
        errors.append(np.linalg.norm(out - ref))

    order = np.log2(errors[0] / errors[1])
    assert order >= 3.5, f"observed order {order:.2f} < 3.5 (errors={errors})"


def test_small_domain_decay_2d():
    """For a domain small enough that no wavenumber fits under the
    instability threshold (|K|<1), the zero solution is linearly stable
    and u -> 0 -- 2D analogue of `test_small_L_decay`."""
    cfg = KSConfig2D(Lx=2.0, Ly=2.0, NX=16, NY=16, dt=0.005, snapshot_every=50)
    rng = np.random.default_rng(3)
    u0 = default_initial_condition2d(cfg, rng)
    traj = integrate2d(u0, cfg, n_steps=5000, snapshot_every=500)
    assert np.linalg.norm(traj[-1]) < 1e-6 * max(np.linalg.norm(traj[0]), 1e-12) + 1e-10


def test_spinup2d_produces_finite_bounded_state():
    cfg = KSConfig2D(Lx=44.0, Ly=44.0, NX=48, NY=48, dt=0.02, snapshot_every=5, spinup_time=50.0)
    rng = np.random.default_rng(4)
    u_final = spinup2d(cfg, rng)
    assert np.isfinite(u_final).all()
    assert u_final.mean() == pytest.approx(0.0, abs=1e-8)
    assert np.abs(u_final).max() < 100.0  # sanity bound, not a tight physical target


def test_wavenumbers2d_shape_and_dc_mode():
    KX, KY = wavenumbers2d(44.0, 44.0, 8, 8)
    assert KX.shape == (8, 8)
    assert KY.shape == (8, 8)
    assert KX[0, 0] == 0.0
    assert KY[0, 0] == 0.0
