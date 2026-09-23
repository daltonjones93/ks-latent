"""Persistent homology validation (brief §6.3): known topology (circle,
torus, 2-sphere), each with and without outliers, to show DTM "earning its
keep" over plain Rips.
"""

from __future__ import annotations

import numpy as np
import pytest

from ks_latent.analysis.dimension import sample_sphere, sample_torus
from ks_latent.analysis.topology import dtm_persistence, lifetimes, max_lifetime, rips_persistence


def _sample_circle(n: int, rng: np.random.Generator) -> np.ndarray:
    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


@pytest.mark.integration
def test_rips_circle_has_one_persistent_h1():
    rng = np.random.default_rng(0)
    X = _sample_circle(200, rng)
    result = rips_persistence(X, homology_dimensions=(0, 1))
    h1 = np.sort(lifetimes(result.diagrams[1]))[::-1]
    assert len(h1) >= 1
    assert h1[0] > 1.0  # clearly off the diagonal
    if len(h1) > 1:
        assert h1[0] > 5 * h1[1]  # dominant over any other H1 feature


@pytest.mark.integration
def test_rips_torus_has_two_persistent_h1_and_one_h2():
    rng = np.random.default_rng(0)
    X = sample_torus(2, 250, rng)
    result = rips_persistence(X, homology_dimensions=(0, 1, 2))
    h1 = np.sort(lifetimes(result.diagrams[1]))[::-1]
    h2 = np.sort(lifetimes(result.diagrams[2]))[::-1]

    assert h1[0] > 1.0 and h1[1] > 1.0  # two independent loops
    if len(h1) > 2:
        assert h1[1] > 2 * h1[2]  # gap after the second loop
    assert h2[0] > 0.5  # the torus's fundamental class
    if len(h2) > 1:
        assert h2[0] > 5 * h2[1]


@pytest.mark.integration
def test_rips_2sphere_has_one_persistent_h2_and_trivial_h1():
    rng = np.random.default_rng(0)
    X = sample_sphere(2, 250, rng)
    result = rips_persistence(X, homology_dimensions=(0, 1, 2))
    assert len(result.diagrams[2]) == 1  # exactly one H2 class
    assert max_lifetime(result.diagrams[2]) > 0.5
    assert max_lifetime(result.diagrams[1]) < max_lifetime(result.diagrams[2])


@pytest.mark.integration
def test_dtm_stable_across_mass_sweep_on_outlier_corrupted_circle():
    """brief §6.3: "assert the conclusion is stable across the sweep"."""
    rng = np.random.default_rng(3)
    X = _sample_circle(150, rng)
    outliers = rng.uniform(-1.5, 1.5, size=(25, 2))
    X_corrupted = np.concatenate([X, outliers], axis=0)

    top_lifetimes = []
    for mass in [0.02, 0.05, 0.1]:
        result = dtm_persistence(X_corrupted, mass=mass, homology_dimensions=(0, 1))
        top = max_lifetime(result.diagrams[1])
        assert top > 0.5, f"DTM at mass={mass} failed to find the loop (lifetime={top:.3f})"
        top_lifetimes.append(top)

    mean = np.mean(top_lifetimes)
    assert all(abs(t - mean) / mean < 0.3 for t in top_lifetimes), (
        f"DTM conclusion not stable across the mass sweep: {top_lifetimes}"
    )


@pytest.mark.integration
def test_dtm_outperforms_plain_rips_on_scattered_outliers():
    """DTM "earning its keep" (brief §6.3): with individually scattered
    outliers (not a separate far-away cluster, which plain Rips handles
    fine on its own), DTM should recover a *cleaner* (larger, more clearly
    dominant) loop than plain Rips on the same corrupted point cloud."""
    rng = np.random.default_rng(3)
    X = _sample_circle(150, rng)
    outliers = rng.uniform(-1.5, 1.5, size=(25, 2))
    X_corrupted = np.concatenate([X, outliers], axis=0)

    plain = rips_persistence(X_corrupted, homology_dimensions=(0, 1))
    dtm = dtm_persistence(X_corrupted, mass=0.05, homology_dimensions=(0, 1))

    assert max_lifetime(dtm.diagrams[1]) > max_lifetime(plain.diagrams[1])
