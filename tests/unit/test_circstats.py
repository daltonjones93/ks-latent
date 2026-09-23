"""Tests for circular statistics (brief §8 D1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ks_latent.utils.circstats import circular_distance, circular_weighted_stats


def test_circular_mean_handles_wraparound():
    """Mass split across the wrap point (x=0 and x=L-1) must average to
    *near the wrap point*, not to the middle of the domain (the naive-mean
    failure mode this module exists to avoid)."""
    L, NX = 100.0, 100
    x = np.arange(NX) * L / NX
    weights = np.zeros(NX)
    weights[0] = 1.0
    weights[-1] = 1.0
    stats = circular_weighted_stats(x, weights, L)
    # Correct answer is near x=99.5 (mod 100) -> equivalently near -0.5 -> 99.5
    assert stats.centroid == pytest.approx(99.5, abs=1e-6)
    assert stats.resultant_length > 0.99  # tightly concentrated


def test_circular_stats_uniform_weights_has_low_resultant_length():
    L, NX = 100.0, 100
    x = np.arange(NX) * L / NX
    weights = np.ones(NX)
    stats = circular_weighted_stats(x, weights, L)
    assert stats.resultant_length < 0.01
    assert stats.spread > 3.0  # large spread for near-uniform


def test_circular_stats_single_delta_has_zero_spread():
    L, NX = 100.0, 100
    x = np.arange(NX) * L / NX
    weights = np.zeros(NX)
    weights[42] = 1.0
    stats = circular_weighted_stats(x, weights, L)
    assert stats.centroid == pytest.approx(x[42], abs=1e-6)
    assert stats.spread < 1e-6


def test_circular_weighted_stats_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        circular_weighted_stats(np.array([0.0, 1.0]), np.array([1.0, -1.0]), L=2.0)


def test_circular_weighted_stats_all_zero_weights_is_a_reportable_degenerate_state():
    """Added 2026-09-07: a genuinely dead/zero-sensitivity latent channel
    (e.g. encoder_kind="spectral_field"'s always-zero DC-imaginary slot)
    must be reported, not crash the whole diagnostic -- see
    circular_weighted_stats's docstring."""
    L, NX = 100.0, 100
    x = np.arange(NX) * L / NX
    weights = np.zeros(NX)
    stats = circular_weighted_stats(x, weights, L)
    assert stats.resultant_length == 0.0
    assert math.isnan(stats.centroid)
    assert math.isinf(stats.spread)


def test_circular_distance_wraps():
    assert circular_distance(0, 9, 10) == 1
    assert circular_distance(2, 7, 10) == 5
    assert circular_distance(np.array([0, 1]), np.array([9, 1]), 10).tolist() == [1, 0]
