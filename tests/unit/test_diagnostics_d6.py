"""Validation for D6 (temporal coherence structure -- user-directed
2026-08-30, not part of the original brief; see
docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 8 and
`ks_latent.analysis.diagnostics`'s D6 module docstring for the full
motivation: D3's Jacobian-bandedness is blind to a broadband-but-still-
ring-respecting operator like an FNO layer's spectral conv, so this
diagnostic instead looks for emergent spatial structure directly in the
raw encoded trajectory via lagged, windowed cross-correlation between
channels -- a statistic the L_decorr regularizer's same-time covariance
constraint does not touch).
"""

from __future__ import annotations

import numpy as np
import pytest

from ks_latent.analysis.diagnostics import (
    laplacian_eigenmap,
    local_temporal_coherence,
    temporal_coherence_diagnostic,
)


def _ring_coherent_trajectory(
    d: int, T: int, shift_per_channel: int, ar_phi: float, noise_std: float, rng: np.random.Generator
) -> np.ndarray:
    """Synthetic `(T, d)` trajectory where channel `i` is a delayed copy
    (`i * shift_per_channel` samples) of one shared AR(1) process (`x[t] =
    ar_phi * x[t-1] + eps[t]`, correlation length `~= 1/(1-ar_phi)`), plus
    noise -- a ground-truth ring structure (ring distance in channel index
    == difference in delay) whose correlation genuinely DECAYS with delay
    difference (unlike a sum of pure sinusoids, which stays ~perfectly
    autocorrelated at any lag once a search covers it -- an AR(1) process's
    finite memory is what makes "ring-adjacent, small delay difference"
    pairs distinguishable from "far, large delay difference" pairs at all)."""
    n_extra = d * shift_per_channel + 200
    n = T + n_extra
    base = np.empty(n)
    base[0] = rng.standard_normal()
    innovations = rng.standard_normal(n - 1)
    for t in range(1, n):
        base[t] = ar_phi * base[t - 1] + innovations[t - 1]
    Z = np.zeros((T, d))
    for i in range(d):
        delay = i * shift_per_channel
        Z[:, i] = base[delay : delay + T] + noise_std * rng.standard_normal(T)
    return Z


def test_window_must_exceed_twice_max_lag():
    Z = np.random.default_rng(0).standard_normal((200, 5))
    with pytest.raises(ValueError, match="window"):
        local_temporal_coherence(Z, window=4, max_lag=3)


def test_local_temporal_coherence_is_symmetric():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((300, 6))
    C = local_temporal_coherence(Z, window=40, max_lag=3)
    assert C.shape == (6, 6)
    np.testing.assert_allclose(C, C.T)


def test_local_temporal_coherence_diagonal_is_near_one():
    """A channel is maximally coherent with itself at zero lag."""
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((300, 5))
    C = local_temporal_coherence(Z, window=40, max_lag=2)
    assert np.all(np.diagonal(C) > 0.99)


def test_local_temporal_coherence_accepts_multiple_trajectories():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((3, 300, 5))  # (n_runs, T, d)
    C = local_temporal_coherence(Z, window=40, max_lag=2)
    assert C.shape == (5, 5)


def test_local_temporal_coherence_detects_ring_neighbours_over_far_pairs():
    """Ring-adjacent channels (small index difference, small true delay
    difference, `shift_per_channel=4`) must show higher coherence than
    far-apart channels (delay difference `5x` larger) -- the core claim
    this diagnostic is built on. `max_lag=6` covers a neighbour's delay (4)
    but not a far pair's (20), so the AR(1) base signal's finite memory
    (correlation length `~1/(1-0.85) ~= 6.7` samples) is what must show up
    as a real gap, not the lag search alone compensating for both."""
    rng = np.random.default_rng(0)
    Z = _ring_coherent_trajectory(d=10, T=4000, shift_per_channel=4, ar_phi=0.85, noise_std=0.1, rng=rng)
    C = local_temporal_coherence(Z, window=60, max_lag=6)
    neighbour_coherence = np.mean([C[i, (i + 1) % 10] for i in range(10)])
    far_coherence = np.mean([C[i, (i + 5) % 10] for i in range(10)])
    assert neighbour_coherence > far_coherence


def test_laplacian_eigenmap_shape_and_orthogonality():
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 1, size=(8, 8))
    A = (A + A.T) / 2
    emb = laplacian_eigenmap(A, n_components=3)
    assert emb.shape == (8, 3)
    # Eigenvectors of a real symmetric matrix are orthonormal.
    np.testing.assert_allclose(emb.T @ emb, np.eye(3), atol=1e-8)


def test_laplacian_eigenmap_caps_n_components_below_d():
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 1, size=(4, 4))
    A = (A + A.T) / 2
    emb = laplacian_eigenmap(A, n_components=10)
    assert emb.shape == (4, 3)  # capped at d - 1


def test_temporal_coherence_diagnostic_finds_significant_bandedness_on_ring_structure():
    rng = np.random.default_rng(1)
    Z = _ring_coherent_trajectory(d=12, T=4000, shift_per_channel=3, ar_phi=0.85, noise_std=0.15, rng=rng)
    result = temporal_coherence_diagnostic(Z, window=100, max_lag=4, n_null=200, seed=0)
    assert result.C.shape == (12, 12)
    assert result.embedding.shape == (12, 2)
    assert result.bandedness_p_value < 0.05


def test_temporal_coherence_diagnostic_independent_channels_not_significant():
    """`window` must be >> the number of lags scanned (`2*max_lag+1`) for
    well-calibrated p-values: `max` over many lags of a correlation
    estimated from too few samples is a multiple-comparisons problem with a
    real positive bias (verified empirically: `window=50, max_lag=10` gives
    a systematically-inflated ~0.3 baseline coherence for pure independent
    noise and a false-positive rate well above the nominal 5%; `window=200,
    max_lag=5` does not) -- this is a property of the estimator, not a bug
    in `bandedness_p_value` (reused verbatim from D3, already relied on
    there). Document the guidance here rather than silently picking
    "safe" numbers with no explanation."""
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((3000, 12))  # fully independent channels, no structure at all
    result = temporal_coherence_diagnostic(Z, window=200, max_lag=5, n_null=200, seed=0)
    assert result.bandedness_p_value > 0.2
