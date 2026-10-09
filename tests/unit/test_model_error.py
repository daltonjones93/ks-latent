"""Tests for ks_latent/da/model_error.py (task I22)."""
import numpy as np
import pytest
import torch

from ks_latent.da.model_error import estimate_site_model_error, sample_site_noise


def _q(c, seed):
    A = np.random.default_rng(seed).standard_normal((c, c))
    return A @ A.T + 0.5 * np.eye(c)


def test_estimate_recovers_known_bias_and_covariance():
    nx, ny, c, T = 6, 5, 3, 4000
    Q = _q(c, 0); b = np.array([0.3, -0.1, 0.05])
    rng = np.random.default_rng(1)
    R = (rng.standard_normal((T, nx * ny, c)) @ np.linalg.cholesky(Q).T + b).reshape(T, -1)
    est = estimate_site_model_error(R, nx, ny, c)
    np.testing.assert_allclose(est.bias, b, atol=0.02)
    np.testing.assert_allclose(est.Q, Q, rtol=0.03, atol=0.02)
    assert est.neighbour_corr < 0.02  # independent sites
    assert est.n_samples == T * nx * ny


def test_neighbour_correlation_detects_spatially_correlated_residuals():
    nx, ny, c, T = 8, 8, 2, 500
    rng = np.random.default_rng(2)
    base = rng.standard_normal((T, nx, ny, c))
    smooth = base + np.roll(base, 1, axis=1) + np.roll(base, 1, axis=2)  # neighbours share draws
    est = estimate_site_model_error(smooth.reshape(T, -1), nx, ny, c)
    assert est.neighbour_corr > 0.25


def test_sample_site_noise_has_target_covariance_and_layout():
    c, n_sites = 4, 9
    Q = _q(c, 3)
    xi = sample_site_noise(20000, n_sites, Q, scale=2.0, generator=torch.Generator().manual_seed(0))
    assert xi.shape == (20000, n_sites * c) and xi.dtype == torch.float32
    per_site = xi.reshape(20000, n_sites, c).double().numpy()
    emp = np.cov(per_site[:, 4, :].T)
    np.testing.assert_allclose(emp, 2.0 * Q, rtol=0.05, atol=0.05)
    cross = np.mean(per_site[:, 0, 0] * per_site[:, 1, 0])
    assert abs(cross) < 0.05 * 2.0 * Q[0, 0]


def test_estimate_raises_on_degenerate_residuals():
    R = np.zeros((10, 4 * 2))
    with pytest.raises(np.linalg.LinAlgError):
        estimate_site_model_error(R, 2, 2, 2)
