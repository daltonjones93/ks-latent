"""Validation for D5 (local intrinsic dimension vs. patch length), brief §8.

Synthetic "extensive" system: NX total dims split into independent chunks,
each chunk generated from a small d0-dim latent via a random linear map --
by construction the *local dimension density* is exactly `d0/chunk_size`,
so a patch spanning `n` whole chunks should read `~ n*d0`, giving a known
target slope `d0/chunk_size` to check the fit against.
"""

from __future__ import annotations

import numpy as np
import pytest

from ks_latent.analysis.diagnostics import extract_patches, local_dimension_vs_length


def _synthetic_extensive_points(n_points: int, n_chunks: int, chunk_size: int, d0: int, rng) -> np.ndarray:
    NX = n_chunks * chunk_size
    points = np.empty((n_points, NX))
    for c in range(n_chunks):
        latent = rng.normal(size=(n_points, d0))
        projection = rng.normal(size=(d0, chunk_size))
        points[:, c * chunk_size : (c + 1) * chunk_size] = latent @ projection
    return points


def test_extract_patches_fixed_start_and_wraparound():
    points = np.arange(20).reshape(1, 20).astype(float)
    patch = extract_patches(points, length=5, rng=np.random.default_rng(0), start=18)
    assert patch[0].tolist() == [18.0, 19.0, 0.0, 1.0, 2.0]  # wraps around


def test_d5_recovers_known_extensive_slope():
    rng = np.random.default_rng(0)
    chunk_size, d0 = 10, 2
    n_chunks = 8
    points = _synthetic_extensive_points(3000, n_chunks, chunk_size, d0, rng)

    # Lengths that are whole multiples of chunk_size, so the patch always
    # spans an integer number of independent chunks -- a clean test of the
    # slope-fitting machinery, isolated from patch/chunk-boundary effects.
    lengths = [chunk_size * k for k in [1, 2, 3, 4, 5]]
    result = local_dimension_vs_length(points, lengths, rng)

    true_slope = d0 / chunk_size  # = 0.2
    # Two-NN's own well-documented downward bias (docs/OPEN_QUESTIONS.md)
    # means the CI need not bracket the true slope exactly -- check the
    # point estimate is close and the CI is a sane, narrow interval.
    assert result.slope == pytest.approx(true_slope, rel=0.25)
    lo, hi = result.slope_ci
    assert lo < hi
    assert abs(true_slope - result.slope) < 3 * (hi - lo)  # true value not wildly outside the CI's scale
