"""Validation for D3 (Jacobian coupling graph, seriation, bandedness
p-value), brief §8."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from ks_latent.analysis.diagnostics import (
    _distance_correlation,
    bandedness,
    bandedness_p_value,
    coupling_graph_diagnostic,
    jacobian_coupling,
)


def _circulant_banded_matrix(d: int, half_width: int, rng: np.random.Generator) -> np.ndarray:
    A = np.zeros((d, d))
    for i in range(d):
        for w in range(-half_width, half_width + 1):
            j = (i + w) % d
            A[i, j] = rng.uniform(0.5, 1.5)
    return A


def test_jacobian_coupling_recovers_known_banded_linear_map():
    torch.manual_seed(0)
    d = 10
    rng = np.random.default_rng(0)
    A_true = _circulant_banded_matrix(d, half_width=1, rng=rng)
    A_true_t = torch.tensor(A_true, dtype=torch.float32)

    def step(z_prev, z_curr):
        return z_curr @ A_true_t.T

    z_prev = torch.randn(50, d)
    z_curr = torch.randn(50, d)
    A_measured = jacobian_coupling(step, z_prev, z_curr)
    assert np.allclose(A_measured, np.abs(A_true), atol=1e-5)


def test_bandedness_p_value_significant_for_banded_matrix():
    rng = np.random.default_rng(1)
    d = 12
    A = _circulant_banded_matrix(d, half_width=1, rng=rng)
    _, observed, p_value = bandedness_p_value(A, n_null=1000, seed=2)
    assert p_value < 0.01, f"expected a strongly banded matrix to score significantly, got p={p_value}"


def test_bandedness_p_value_not_significant_for_dense_random_matrix():
    rng = np.random.default_rng(3)
    d = 12
    A = rng.uniform(0.0, 1.0, size=(d, d))  # no structure at all
    _, observed, p_value = bandedness_p_value(A, n_null=1000, seed=4)
    assert p_value > 0.05, f"expected a random dense matrix to score unremarkably, got p={p_value}"


def test_bandedness_recovers_perfect_score_for_identity_permutation_on_diagonal_matrix():
    d = 6
    A = np.eye(d)
    perm = np.arange(d)
    score = bandedness(A, perm, bandwidth=3.0)
    assert score == pytest.approx(1.0, abs=1e-6)  # every mass entry at distance 0, w(0)=1


def test_distance_correlation_detects_nonlinear_dependence():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 2000)
    y_dependent = x**2 + 0.01 * rng.normal(size=2000)  # nonlinear but dependent
    y_independent = rng.normal(size=2000)
    dcor_dep = _distance_correlation(x, y_dependent)
    dcor_indep = _distance_correlation(x, y_independent)
    # y=x^2 with x symmetric about 0 is a classic case where dCor is only
    # moderate (~0.4-0.5), not close to 1, because the even symmetry
    # cancels much of the linear-scale signal dCor is sensitive to -- the
    # real test is a clear separation from the independent case, not an
    # absolute threshold near 1.
    assert dcor_dep > 0.35
    assert dcor_indep < 0.1
    assert dcor_dep > 3 * dcor_indep


def test_coupling_graph_diagnostic_end_to_end_shapes():
    torch.manual_seed(0)
    d = 6
    rng = np.random.default_rng(0)
    A_true = _circulant_banded_matrix(d, half_width=1, rng=rng)
    A_true_t = torch.tensor(A_true, dtype=torch.float32)

    def step(z_prev, z_curr):
        return torch.tanh(z_curr @ A_true_t.T)

    z_prev = torch.randn(100, d)
    z_curr = torch.randn(100, d)
    result = coupling_graph_diagnostic(step, z_prev, z_curr, n_null=200, seed=5)
    assert result.A_jacobian.shape == (d, d)
    assert result.A_distance_corr.shape == (d, d)
    assert result.A_mutual_info.shape == (d, d)
    assert result.permutation.shape == (d,)
    assert 0.0 <= result.bandedness_p_value <= 1.0
