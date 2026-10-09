"""Tests for the physical-space local ETKF (task I11, ks_latent/da/letkf_phys.py)."""
import numpy as np
import pytest

from ks_latent.da.letkf_phys import LocalETKFPhys2D


def _setup(N=20, NX=8, s=2, seed=0):
    rng = np.random.default_rng(seed)
    n = NX * NX
    Xf = rng.standard_normal((N, n)) @ np.diag(rng.uniform(0.5, 2.0, n)) + rng.standard_normal(n)
    g = np.arange(0, NX, s)
    obs_xy = np.array([(i, j) for i in g for j in g])
    obs_idx = obs_xy[:, 0] * NX + obs_xy[:, 1]
    y = rng.standard_normal(obs_idx.size)
    return Xf, obs_xy, obs_idx, y


def test_unlocalized_matches_kalman_mean_and_covariance():
    Xf, obs_xy, obs_idx, y = _setup()
    r = 0.7
    f = LocalETKFPhys2D(8, 8.0, obs_xy, r_std=r, c=None, block=4)
    Xa = f.analyze(Xf, y, obs_idx)
    N = Xf.shape[0]
    xm = Xf.mean(0); Xp = Xf - xm
    Pf = Xp.T @ Xp / (N - 1)
    H = np.zeros((obs_idx.size, Xf.shape[1])); H[np.arange(obs_idx.size), obs_idx] = 1
    K = Pf @ H.T @ np.linalg.inv(H @ Pf @ H.T + r ** 2 * np.eye(obs_idx.size))
    np.testing.assert_allclose(Xa.mean(0), xm + K @ (y - H @ xm), atol=1e-10)
    Ap = Xa - Xa.mean(0)
    np.testing.assert_allclose(Ap.T @ Ap / (N - 1), (np.eye(Pf.shape[0]) - K @ H) @ Pf, atol=1e-9)


def test_localization_ignores_distant_observations():
    """A block whose GC support contains no observation is returned unchanged."""
    Xf, obs_xy, obs_idx, y = _setup(NX=16, s=16)  # one observation at (0, 0)
    f = LocalETKFPhys2D(16, 16.0, obs_xy, r_std=0.5, c=2.0, block=4)  # support radius 4 units
    Xa = f.analyze(Xf, y, obs_idx)
    far = (np.arange(16)[:, None] // 4 == 2) & (np.arange(16)[None, :] // 4 == 2)  # block centred at (9.5, 9.5)
    np.testing.assert_array_equal(Xa[:, far.reshape(-1)], Xf[:, far.reshape(-1)])
    assert not np.allclose(Xa[:, 0], Xf[:, 0])


def test_rtps_full_relaxation_restores_prior_spread():
    Xf, obs_xy, obs_idx, y = _setup()
    f = LocalETKFPhys2D(8, 8.0, obs_xy, r_std=0.3, c=3.0, block=4, rtps=1.0)
    Xa = f.analyze(Xf, y, obs_idx)
    np.testing.assert_allclose(Xa.std(0, ddof=1), Xf.std(0, ddof=1), rtol=1e-10)


def test_rejects_indivisible_block():
    with pytest.raises(ValueError):
        LocalETKFPhys2D(10, 10.0, np.zeros((1, 2), int), r_std=1.0, c=1.0, block=4)
