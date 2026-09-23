"""Validation for D8 (same-time SIGNED channel correlation -- user-directed
2026-09-01, not part of the original brief; see
`ks_latent.analysis.diagnostics`'s D8 module docstring). D7 uses
`|correlation|`, so an anti-correlated near-neighbor scores identically to a
correlated one -- D8 uses SIGNED correlation and asks the sharper question:
does the ACTUAL, fixed latent index order exhibit SAME-SIGN local
coherence, not just *some* coupling.
"""

from __future__ import annotations

import numpy as np

from ks_latent.analysis.diagnostics import (
    same_time_channel_correlation,
    same_time_channel_correlation_signed,
    same_time_coupling_diagnostic,
    same_time_coupling_diagnostic_signed,
    signed_bandedness,
    signed_bandedness_p_value,
)


def _ring_correlated_samples(d: int, n: int, rho: float, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Same construction as test_diagnostics_d7.py's helper -- a ground-
    truth "neighboring channels vary together" (SAME-SIGN) structure."""
    idx = np.arange(d)
    dist = np.minimum(np.abs(idx[:, None] - idx[None, :]), d - np.abs(idx[:, None] - idx[None, :]))
    cov = rho ** dist.astype(float)
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((n, d)) @ L.T
    Z += noise_std * rng.standard_normal((n, d))
    return Z


def test_same_time_channel_correlation_signed_shape_symmetry_and_sign():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((500, 10))
    A = same_time_channel_correlation_signed(Z)
    assert A.shape == (10, 10)
    assert np.allclose(A, A.T)
    assert np.allclose(np.diag(A), 1.0)
    assert (A >= -1.0 - 1e-9).all() and (A <= 1.0 + 1e-9).all()
    assert (A < 0).any()  # unlike D7's matrix, genuinely has negative entries


def test_signed_correlation_magnitude_matches_unsigned():
    """`|signed correlation| == unsigned (D7) correlation` -- the two
    diagnostics measure the same underlying coupling, just retain (D8) or
    discard (D7) its sign."""
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((500, 12))
    signed = same_time_channel_correlation_signed(Z)
    unsigned = same_time_channel_correlation(Z)
    assert np.allclose(np.abs(signed), unsigned)


def test_d8_finds_significant_signed_bandedness_on_same_sign_ring():
    """A genuinely same-sign locally-correlated ring (positive rho) should
    score significant under D8 too, same as it does under D7."""
    rng = np.random.default_rng(2)
    Z = _ring_correlated_samples(d=44, n=6000, rho=0.85, noise_std=0.05, rng=rng)
    result = same_time_coupling_diagnostic_signed(Z, n_null=200, seed=0)
    assert result.A.shape == (44, 44)
    assert result.bandedness_p_value < 0.05
    assert result.bandedness_observed > 0


def test_d8_distinguishes_anti_correlated_local_structure_that_d7_cannot():
    """The key differentiator this whole diagnostic exists for: flip every
    other channel's sign (Z = Y * (-1)**i) on the SAME ring-correlated
    data. |correlation| is invariant to per-channel sign flips, so D7's
    statistic is UNCHANGED -- still strongly significant. But every
    immediate-neighbor pair (dist=1) now has SIGN(i)*SIGN(i+1) = -1, i.e.
    neighbors are systematically ANTI-correlated -- D8 must NOT report
    this as significant same-sign coherence."""
    rng = np.random.default_rng(3)
    d = 44
    Y = _ring_correlated_samples(d=d, n=6000, rho=0.85, noise_std=0.05, rng=rng)
    sign_pattern = np.array([(-1.0) ** i for i in range(d)])
    Z = Y * sign_pattern

    d7_signed_input = same_time_coupling_diagnostic(Z, n_null=200, seed=0)
    d7_original = same_time_coupling_diagnostic(Y, n_null=200, seed=0)
    assert np.allclose(d7_signed_input.A, d7_original.A)  # D7 truly can't tell the difference

    d8_result = same_time_coupling_diagnostic_signed(Z, n_null=200, seed=0)
    assert d8_result.bandedness_p_value > 0.05 or d8_result.bandedness_observed < 0


def test_d8_independent_channels_not_significant():
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((5020, 44))
    result = same_time_coupling_diagnostic_signed(Z, n_null=200, seed=0)
    assert result.bandedness_p_value > 0.05


def test_signed_bandedness_null_is_roughly_calibrated_at_d44_noise():
    """Same style of calibration check as D7's entry-shuffle null
    (test_diagnostics_d7.py) -- random-relabeling null on pure signed
    noise should not show an inflated false-positive rate."""
    n_below_threshold = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((44, 44))
        A = (M + M.T) / 2
        np.fill_diagonal(A, 1.0)
        _, p = signed_bandedness_p_value(A, n_null=200, seed=0)
        if p < 0.05:
            n_below_threshold += 1
    assert n_below_threshold <= 2  # roughly nominal (~5-10%), not inflated


def test_signed_bandedness_is_finite_on_small_batch():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((6, 6))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 1.0)
    score = signed_bandedness(A)
    assert np.isfinite(score)
