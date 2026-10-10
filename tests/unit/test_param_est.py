"""Tests for ks_latent/da/param_est.py (task J16)."""
import numpy as np
import pytest

from ks_latent.da.param_est import global_param_update


def _ens(N=40, m=25, seed=0):
    rng = np.random.default_rng(seed)
    theta = 1.2 + 0.1 * rng.standard_normal(N)
    # observations correlated with theta plus independent parts
    Hx = np.outer(theta - theta.mean(), rng.standard_normal(m)) + 0.3 * rng.standard_normal((N, m))
    Y = Hx - Hx.mean(0)
    d = rng.standard_normal(m) * 0.2
    return theta, Y, d


def test_mean_equals_ensemble_kalman_regression():
    theta, Y, d = _ens()
    r = 0.4
    N = theta.size
    ta = global_param_update(theta, Y, d, r)
    tp = theta - theta.mean()
    cov_tY = tp @ Y / (N - 1)
    cov_YY = Y.T @ Y / (N - 1)
    expected = theta.mean() + cov_tY @ np.linalg.solve(cov_YY + r ** 2 * np.eye(Y.shape[1]), d)
    assert abs(ta.mean() - expected) < 1e-10


def test_spread_matches_kalman_posterior_variance_without_safeguards():
    theta, Y, d = _ens(seed=1)
    r = 0.4
    N = theta.size
    ta = global_param_update(theta, Y, d, r)
    tp = theta - theta.mean()
    var_t = tp @ tp / (N - 1)
    cov_tY = tp @ Y / (N - 1)
    cov_YY = Y.T @ Y / (N - 1)
    post = var_t - cov_tY @ np.linalg.solve(cov_YY + r ** 2 * np.eye(Y.shape[1]), cov_tY)
    assert abs(ta.var(ddof=1) - post) < 1e-10


def test_rtps_one_restores_prior_spread_and_floor_is_respected():
    theta, Y, d = _ens(seed=2)
    ta = global_param_update(theta, Y, d, 0.1, rtps=1.0)
    assert abs(ta.std(ddof=1) - theta.std(ddof=1)) < 1e-12
    tb = global_param_update(theta, Y, d, 0.01, min_std=0.05)
    assert tb.std(ddof=1) >= 0.05 - 1e-12


def test_no_information_leaves_mean_unchanged():
    theta, Y, d = _ens(seed=3)
    ta = global_param_update(theta, Y, np.zeros_like(d), 0.5)
    assert abs(ta.mean() - theta.mean()) < 1e-12


def test_shape_check():
    with pytest.raises(ValueError):
        global_param_update(np.zeros(5), np.zeros((4, 3)), np.zeros(3), 1.0)


def test_joint_update_columns_equal_scalar_updates_and_joint_regression():
    from ks_latent.da.param_est import global_params_update
    theta, Y, d = _ens(seed=4)
    rng = np.random.default_rng(9)
    Theta = np.stack([theta, 2.0 + 0.3 * rng.standard_normal(theta.size), 0.5 * theta + 0.1 * rng.standard_normal(theta.size)], 1)
    r = 0.4
    Ta = global_params_update(Theta, Y, d, r)
    for j in range(3):
        assert np.allclose(Ta[:, j], global_param_update(Theta[:, j], Y, d, r), atol=1e-12)
    N = Theta.shape[0]
    Tp = Theta - Theta.mean(0)
    K = (Tp.T @ Y / (N - 1)) @ np.linalg.inv(Y.T @ Y / (N - 1) + r ** 2 * np.eye(Y.shape[1]))
    assert np.allclose(Ta.mean(0), Theta.mean(0) + K @ d, atol=1e-10)
    Tb = global_params_update(Theta, Y, d, 0.01, min_std=np.array([0.05, 0.2, 0.0]))
    assert Tb[:, 0].std(ddof=1) >= 0.05 - 1e-12 and Tb[:, 1].std(ddof=1) >= 0.2 - 1e-12
