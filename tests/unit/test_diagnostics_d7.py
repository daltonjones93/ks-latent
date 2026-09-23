"""Validation for D7 (same-time channel correlation -- user-directed
2026-08-31, not part of the original brief; see
docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 22 and
`ks_latent.analysis.diagnostics`'s D7 module docstring). Unlike D6
(temporal, lagged), D7 is the genuinely SAME-INSTANT statistic: does
`encode(u(t))_i` covary with `encode(u(t))_j` across real samples at a
single time, for `i`, `j` near each other under some permutation -- the
most literal reading of "do neighboring latent variables vary together,"
requiring no propagator and no time lag at all.
"""

from __future__ import annotations

import numpy as np

from ks_latent.analysis.diagnostics import (
    bandedness_p_value,
    bandedness_p_value_entry_shuffle,
    laplacian_eigenmap,
    same_time_channel_correlation,
    same_time_coupling_diagnostic,
)


def _ring_correlated_samples(d: int, n: int, rho: float, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """`(n, d)` i.i.d. SAME-TIME samples with a circulant covariance
    (`Cov[i,j] = rho^circular_distance(i,j)`, via an AR-like construction
    around a ring) -- a ground-truth "neighboring channels vary together"
    structure with no time dimension at all (every row is an independent
    draw, not a trajectory)."""
    idx = np.arange(d)
    dist = np.minimum(np.abs(idx[:, None] - idx[None, :]), d - np.abs(idx[:, None] - idx[None, :]))
    cov = rho ** dist.astype(float)
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((n, d)) @ L.T
    Z += noise_std * rng.standard_normal((n, d))
    return Z


def test_same_time_channel_correlation_shape_and_symmetry():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((500, 10))
    A = same_time_channel_correlation(Z)
    assert A.shape == (10, 10)
    assert np.allclose(A, A.T)
    assert np.allclose(np.diag(A), 1.0)


def test_same_time_channel_correlation_is_nonnegative():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((500, 10))
    A = same_time_channel_correlation(Z)
    assert (A >= 0).all()


def test_diagnostic_finds_significant_bandedness_on_planted_ring_structure():
    """Power is `d`-dependent for this bandwidth (verified empirically,
    2026-08-31): weak/inconsistent at `d=12` (the Gaussian weight's
    `bandwidth=3` still gives non-negligible weight to "far" pairs on a
    small ring), clean at `d=44` (this project's actual latent dimension,
    where "far" is more clearly separated from "near") -- so this test uses
    `d=44` to match real usage rather than the smaller `d=12` scale D6's
    analogous test uses."""
    rng = np.random.default_rng(1)
    Z = _ring_correlated_samples(d=44, n=6000, rho=0.85, noise_std=0.05, rng=rng)
    result = same_time_coupling_diagnostic(Z, n_null=200, seed=0)
    assert result.A.shape == (44, 44)
    assert result.embedding.shape == (44, 2)
    assert result.bandedness_p_value < 0.05


def test_diagnostic_independent_channels_not_significant():
    """Calibration check (mirroring D6's own, but at `d=44` -- see this
    module's docstring: D6/D3's shared `bandedness_p_value` was found to be
    badly miscalibrated at the real `d=44` scale, only ever checked at
    `d=12` before; `same_time_coupling_diagnostic` uses the corrected
    `bandedness_p_value_entry_shuffle` null instead, verified here)."""
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((5020, 44))  # no planted structure at all, real d/n scale
    result = same_time_coupling_diagnostic(Z, n_null=200, seed=0)
    # A single seed's p-value is a draw from ~Uniform(0,1) under a properly
    # calibrated null, so only check it isn't spuriously "significant" --
    # the actual false-positive RATE is checked with many seeds in
    # test_entry_shuffle_null_is_calibrated_at_d44_noise below.
    assert result.bandedness_p_value > 0.05


def test_diagnostic_permutation_recovers_ring_neighbours_over_far_pairs():
    """The discovered ordering should put ring-adjacent channels closer
    together (in the FOUND order) than channels that were far apart on the
    true ring, on average -- not just a significant p-value, but a
    permutation that is actually usable."""
    rng = np.random.default_rng(3)
    d = 44
    Z = _ring_correlated_samples(d=d, n=6000, rho=0.85, noise_std=0.05, rng=rng)
    result = same_time_coupling_diagnostic(Z, n_null=200, seed=0)
    pos = np.argsort(result.permutation)  # position of each original channel in the found order
    true_neighbour_gaps, far_pair_gaps = [], []
    for i in range(d):
        j_near = (i + 1) % d
        j_far = (i + d // 2) % d
        gap_near = min(abs(pos[i] - pos[j_near]), d - abs(pos[i] - pos[j_near]))
        gap_far = min(abs(pos[i] - pos[j_far]), d - abs(pos[i] - pos[j_far]))
        true_neighbour_gaps.append(gap_near)
        far_pair_gaps.append(gap_far)
    assert np.mean(true_neighbour_gaps) < np.mean(far_pair_gaps)


def test_original_bandedness_p_value_is_miscalibrated_at_d44_noise():
    """Documents the bug this file's fix (`bandedness_p_value_entry_shuffle`)
    exists for: the ORIGINAL `bandedness_p_value` (shared by D3/D6, still
    unfixed there) systematically under-reports p-values on pure noise at
    the real `d=44` scale, because its null (random relabelings of the
    SAME matrix) doesn't correct for the Fiedler vector being specifically
    optimized to concentrate that matrix's largest entries near the
    diagonal -- an adaptive-search bias, not a property of genuine
    structure. This is a regression guard: if `bandedness_p_value` is ever
    "fixed" in a way that changes this behavior, this test should be
    revisited (it is not asserting correctness, only documenting the
    known-bad status quo)."""
    n_below_threshold = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((44, 44))
        A = np.abs((M + M.T) / 2)
        np.fill_diagonal(A, 0)
        _, _, p = bandedness_p_value(A, n_null=200, seed=0)
        if p < 0.05:
            n_below_threshold += 1
    assert n_below_threshold >= 4  # badly inflated false-positive rate (nominal would be ~0-1/8)


def test_entry_shuffle_null_is_calibrated_at_d44_noise():
    """The fix: `bandedness_p_value_entry_shuffle` on the same kind of pure
    noise should NOT show an inflated false-positive rate."""
    n_below_threshold = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((44, 44))
        A = np.abs((M + M.T) / 2)
        np.fill_diagonal(A, 0)
        _, _, p = bandedness_p_value_entry_shuffle(A, n_null=200, seed=0)
        if p < 0.05:
            n_below_threshold += 1
    assert n_below_threshold <= 2  # roughly nominal (~5-10%), not inflated
