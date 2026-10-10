"""Tests for the modified-coefficient 2D KS operator (task I28)."""
import numpy as np

from ks_latent.config import KSConfig2D
from ks_latent.solver.ks2d import integrate2d, modified_operator2d


def _cfg():
    return KSConfig2D(Lx=22.0, Ly=22.0, NX=32, NY=32, dt=0.02, snapshot_every=10, spinup_time=0.0, seed=0)


def test_lap_coef_one_reproduces_standard_solver():
    """Same operator up to last-bit floating-point differences (~1e-17 observed when run alongside
    other solver tests)."""
    cfg = _cfg()
    u0 = np.random.default_rng(0).standard_normal((32, 32)) * 0.1
    a = integrate2d(u0, cfg, 100)
    b = integrate2d(u0, cfg, 100, op=modified_operator2d(cfg, 1.0))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-13)


def test_most_unstable_wavenumber_moves_to_sqrt_half_coef():
    cfg = _cfg()
    for a in (1.0, 1.3):
        op = modified_operator2d(cfg, a)
        K = np.sqrt(op.KX ** 2 + op.KY ** 2).ravel()
        k = np.linspace(0.05, 2, 4001)
        k_star = k[np.argmax(a * k ** 2 - k ** 4)]
        assert abs(k_star - np.sqrt(a / 2)) < 1e-3
        # the operator's symbol is exactly a K^2 - K^4 on the grid
        np.testing.assert_allclose(op.Lhat.ravel().real, a * K ** 2 - K ** 4, atol=1e-12)


def test_modified_truth_differs_from_standard():
    cfg = _cfg()
    u0 = np.random.default_rng(1).standard_normal((32, 32)) * 0.1
    a = integrate2d(u0, cfg, 200)
    b = integrate2d(u0, cfg, 200, op=modified_operator2d(cfg, 1.2))
    assert np.abs(a[-1] - b[-1]).max() > 1e-3


def test_uniform_advection_is_an_exact_translation():
    """c * dt * n_steps = an integer number of grid points: the advected solution equals the
    unadvected one rolled by that many points (KS commutes with translations)."""
    cfg = _cfg()  # L = 22, NX = 32: dx = 0.6875
    dx = cfg.Lx / cfg.NX
    n = 100
    c = 2 * dx / (n * cfg.dt)  # moves exactly 2 grid points in n steps
    u0 = np.random.default_rng(2).standard_normal((32, 32)) * 0.1
    a = integrate2d(u0, cfg, n, snapshot_every=n)[-1]
    b = integrate2d(u0, cfg, n, snapshot_every=n, advect=(c, 0.0))[-1]
    np.testing.assert_allclose(b, np.roll(a, 2, axis=0), atol=1e-10)


def test_forcing_zero_is_standard_and_nonzero_changes_and_both_rejected():
    import pytest
    from ks_latent.solver.twin import terrain_forcing
    cfg = _cfg()
    u0 = np.random.default_rng(3).standard_normal((32, 32)) * 0.1
    a = integrate2d(u0, cfg, 50)
    np.testing.assert_array_equal(a, integrate2d(u0, cfg, 50, forcing=np.zeros((32, 32))))
    f = terrain_forcing(cfg, 0.5)
    assert abs(f.mean()) < 1e-12 and abs(f.std() - 0.5) < 1e-12
    b = integrate2d(u0, cfg, 50, forcing=f)
    assert np.abs(a[-1] - b[-1]).max() > 1e-3
    with pytest.raises(ValueError):
        integrate2d(u0, cfg, 5, advect=(1.0, 0.0), forcing=f)


def test_pde_filter_step_batch_matches_solver_with_model_errors():
    """exp_i11's batched step with a spurious forcing / advection equals ks2d.integrate2d with the
    same term, member by member (so the 'wrong model' is exactly the documented equation)."""
    import importlib.util
    import pathlib
    import scipy.fft as sfft
    from ks_latent.solver.ks2d import get_operator2d
    from ks_latent.solver.twin import terrain_forcing
    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "exp_i11_pde_letkf.py"
    spec = importlib.util.spec_from_file_location("exp_i11", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg = _cfg()
    op = get_operator2d(cfg)
    u0 = np.random.default_rng(4).standard_normal((2, 32, 32)) * 0.1
    u0 -= u0.mean(axis=(1, 2), keepdims=True)
    f = terrain_forcing(cfg, 0.3)
    V = sfft.fft2(u0)
    for _ in range(20):
        V = mod.step_batch(V, op, np.fft.fft2(f))
    ref = np.stack([integrate2d(u0[i], cfg, 20, snapshot_every=20, forcing=f)[-1] for i in range(2)])
    np.testing.assert_allclose(sfft.ifft2(V).real, ref, atol=1e-10)
    shift = np.exp(-1j * 1.3 * op.KX * cfg.dt)
    V = sfft.fft2(u0)
    for _ in range(20):
        V = mod.step_batch(V, op, None, shift)
    ref = np.stack([integrate2d(u0[i], cfg, 20, snapshot_every=20, advect=(1.3, 0.0))[-1] for i in range(2)])
    np.testing.assert_allclose(sfft.ifft2(V).real, ref, atol=1e-10)


# --- J17A: the missing cubic process -beta * P[u'^3] (2026-10-10) ---
import pytest  # noqa: E402
from ks_latent.solver.ks2d import cubic_term2d, get_operator2d, step2d, step_with_tangent2d  # noqa: E402


def _cfg_small():
    from ks_latent.config import KSConfig2D
    return KSConfig2D(Lx=22.0, Ly=22.0, NX=32, NY=32, dt=0.02, snapshot_every=5, spinup_time=0.0, seed=0)


def _u_small(seed=0):
    from ks_latent.solver.ks2d import default_initial_condition2d
    return 10.0 * default_initial_condition2d(_cfg_small(), np.random.default_rng(seed))


def test_cubic_zero_equals_standard_solver():
    from ks_latent.solver.ks2d import integrate2d
    cfg, u0 = _cfg_small(), _u_small()
    a = integrate2d(u0, cfg, 50, snapshot_every=50)
    b = integrate2d(u0, cfg, 50, snapshot_every=50, cubic=0.0)
    assert np.array_equal(a, b)


def test_cubic_term_matches_direct_evaluation_and_is_mean_invariant():
    u = _u_small(1)
    beta = 1.3e-3
    up = u - u.mean()
    direct = -beta * (up ** 3 - np.mean(up ** 3))
    got = np.fft.ifft2(cubic_term2d(np.fft.fft2(u), beta)).real
    assert np.max(np.abs(got - direct)) < 1e-13
    shifted = np.fft.ifft2(cubic_term2d(np.fft.fft2(u + 5.0), beta)).real  # u -> u + c invariance
    assert np.max(np.abs(shifted - got)) < 1e-12


def test_cubic_single_mode_decay_matches_analytic():
    # u = A cos(k x) on the zero-mean subspace: P[u^3] = (3/4) A^3 cos(kx) + (1/4) A^3 cos(3kx).
    # With the KS terms switched off (direct check of the explicit term) the projected amplitude ODE
    # for small dt is dA/dt = -(3/4) beta A^3.
    cfg = _cfg_small()
    x = np.arange(cfg.NX) * cfg.Lx / cfg.NX
    A, beta = 2.0, 0.05
    u = np.repeat((A * np.cos(2 * np.pi * x / cfg.Lx))[:, None], cfg.NY, axis=1)
    t = np.fft.ifft2(cubic_term2d(np.fft.fft2(u), beta)).real
    k1 = np.cos(2 * np.pi * x / cfg.Lx)
    proj = (t[:, 0] * k1).sum() / (k1 * k1).sum()
    assert abs(proj - (-0.75 * beta * A ** 3)) < 1e-12


def test_cubic_tangent_matches_finite_difference():
    cfg = _cfg_small()
    op = get_operator2d(cfg)
    beta = 2e-3
    v = np.fft.fft2(_u_small(2)); v[0, 0] = 0
    dv = np.fft.fft2(np.random.default_rng(5).standard_normal((cfg.NX, cfg.NY))); dv[0, 0] = 0
    eps = 1e-6
    fd = (step2d(v + eps * dv, op, cubic=beta) - step2d(v - eps * dv, op, cubic=beta)) / (2 * eps)
    vn, dV = step_with_tangent2d(v, dv[None], op, cubic=beta)
    assert np.allclose(vn, step2d(v, op, cubic=beta), atol=1e-10)
    assert np.max(np.abs(dV[0] - fd)) / np.max(np.abs(fd)) < 1e-6


def test_cubic_cannot_combine_with_advection():
    from ks_latent.solver.ks2d import integrate2d
    with pytest.raises(ValueError):
        integrate2d(_u_small(), _cfg_small(), 2, advect=(1.0, 0.0), cubic=1e-3)


def test_exp_i11_step_batch_matches_solver_with_cubic():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from exp_i11_pde_letkf import step_batch
    cfg = _cfg_small()
    op = get_operator2d(cfg)
    V = np.stack([np.fft.fft2(_u_small(i)) for i in range(3)])
    V[:, 0, 0] = 0
    got = step_batch(V.copy(), op, cubic=1.5e-3)
    for i in range(3):
        assert np.allclose(got[i], step2d(V[i].copy(), op, cubic=1.5e-3), atol=1e-11)
