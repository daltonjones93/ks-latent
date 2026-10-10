"""Tests for ks_latent/solver/ks2d_memory.py (J17 variant B, 2026-10-10)."""
import numpy as np
import pytest
from scipy.linalg import expm

from ks_latent.config import KSConfig2D
from ks_latent.solver.ks2d import default_initial_condition2d, integrate2d
from ks_latent.solver.ks2d_memory import MemoryOperator, MemoryParams, integrate_memory, step_memory


def _cfg():
    return KSConfig2D(Lx=22.0, Ly=22.0, NX=32, NY=32, dt=0.02, snapshot_every=5, spinup_time=0.0, seed=0)


def _u(seed=0, amp=10.0):
    return amp * default_initial_condition2d(_cfg(), np.random.default_rng(seed))


def _mode(cfg, amp, j=2, l=1):
    x = np.arange(cfg.NX) * cfg.Lx / cfg.NX
    X, Y = np.meshgrid(x, x, indexing="ij")
    return amp * np.cos(2 * np.pi * (j * X / cfg.Lx + l * Y / cfg.Ly)), 2 * np.pi * np.hypot(j / cfg.Lx, l / cfg.Ly)


def test_eps_zero_u_equals_standard_solver():
    cfg = _cfg()
    U, V = integrate_memory(_u(), None, cfg, 100, MemoryParams(0.0, 2.0, 0.5), snapshot_every=50)
    ref = integrate2d(_u(), cfg, 100, snapshot_every=50)
    assert np.allclose(U, ref, atol=1e-12)
    assert np.abs(V[-1]).max() > 0  # v is still driven by u


def test_v_free_decay_is_exponential():
    cfg = _cfg()
    v0, _ = _mode(cfg, 1.0)
    U, V = integrate_memory(np.zeros_like(v0), v0, cfg, 100, MemoryParams(0.0, 1.5, 0.7), snapshot_every=100)
    assert np.allclose(V[-1], v0 * np.exp(-100 * cfg.dt / 1.5), atol=1e-12)
    assert np.abs(U).max() == 0


@pytest.mark.parametrize("coupling", ["lap", "identity"])
def test_coupled_linear_mode_matches_matrix_exponential(coupling):
    # tiny single Fourier mode: the nonlinearity is negligible (amplitude^2), so (u, v) obeys the exact
    # 2x2 linear system d/dt [u; v] = [[K^2 - K^4, eps], [kappa c, -1/tau]] [u; v], c = -K^2 or 1
    cfg = _cfg()
    eps, tau, kappa = 0.8, 2.0, 0.6
    u0, K = _mode(cfg, 1e-9)
    c = -K ** 2 if coupling == "lap" else 1.0
    A = np.array([[K ** 2 - K ** 4, eps], [kappa * c, -1.0 / tau]])
    T = 150 * cfg.dt
    amp = expm(A * T) @ np.array([1.0, 0.0])
    U, V = integrate_memory(u0, None, cfg, 150, MemoryParams(eps, tau, kappa, coupling), snapshot_every=150)
    phi = u0 / 1e-9
    assert np.allclose(U[-1] / 1e-9, amp[0] * phi, rtol=0, atol=1e-8 * max(1, abs(amp[0])))
    assert np.allclose(V[-1] / 1e-9, amp[1] * phi, rtol=0, atol=1e-8 * max(1, abs(amp[1])))


def test_batched_step_matches_single():
    cfg = _cfg()
    mop = MemoryOperator(cfg, MemoryParams(0.5, 2.0, 0.3))
    Wu = np.stack([np.fft.fft2(_u(i)) for i in range(3)]); Wu[:, 0, 0] = 0
    Wv = 0.1 * np.stack([np.fft.fft2(_u(i + 5)) for i in range(3)]); Wv[:, 0, 0] = 0
    bu, bv = step_memory(Wu.copy(), Wv.copy(), mop)
    for i in range(3):
        su, sv = step_memory(Wu[i].copy(), Wv[i].copy(), mop)
        assert np.allclose(bu[i], su, atol=1e-11) and np.allclose(bv[i], sv, atol=1e-11)


def test_params_validation():
    with pytest.raises(ValueError):
        MemoryParams(0.1, 0.0, 1.0)
    with pytest.raises(ValueError):
        MemoryParams(0.1, 1.0, 1.0, coupling="grad")


def test_letkf_phys_multifield_hidden_copy_gets_the_same_update():
    from ks_latent.da.letkf_phys import LocalETKFPhys2D
    NX, N = 16, 12
    rng = np.random.default_rng(0)
    Xu = rng.standard_normal((N, NX * NX))
    xg = np.arange(0, NX, 4)
    obs_xy = np.array([(i, j) for i in xg for j in xg])
    obs_idx = obs_xy[:, 0] * NX + obs_xy[:, 1]
    y = rng.standard_normal(obs_idx.size)
    f = LocalETKFPhys2D(NX, 10.0, obs_xy, r_std=0.5, c=3.0, block=4, rtps=0.5)
    single = f.analyze(Xu, y, obs_idx)
    both = f.analyze(np.concatenate([Xu, 2.0 * Xu], axis=1), y, obs_idx)
    assert np.allclose(both[:, :NX * NX], single, atol=1e-12)
    assert np.allclose(both[:, NX * NX:], 2.0 * single, atol=1e-12)
