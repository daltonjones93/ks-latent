"""Diffusion-map validation (brief §6.1: "with spectrum plotting", validated
on d-spheres/tori). Diffusion maps don't return a single "dimension" number
like two-NN/correlation dimension, so validation here checks the known
spectral-degeneracy structure instead: a circle's leading nontrivial
eigenspace is 2-dim (degenerate cos/sin pair), a 2-sphere's is 3-dim
(l=1 spherical harmonics), each separated from the rest by a spectral gap.
Individual eigenvectors within a degenerate eigenspace are only defined up
to an arbitrary rotation (`eigh` returns *some* orthonormal basis of the
eigenspace), so the checks use the rotation-invariant embedding radius
within that eigenspace rather than comparing eigenvectors component-wise to
cos/sin.
"""

from __future__ import annotations

import numpy as np

from ks_latent.analysis.dimension import diffusion_maps, sample_sphere, sample_torus


def test_diffusion_map_circle_has_degenerate_leading_pair():
    rng = np.random.default_rng(0)
    angles = rng.uniform(0.0, 2.0 * np.pi, 1000)
    X = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    result = diffusion_maps(X, n_components=6)

    # Leading pair nearly degenerate, then a clear gap to the next mode.
    assert result.eigenvalues[0] / result.eigenvalues[1] < 1.1
    assert result.eigenvalues[1] / result.eigenvalues[2] > 2.0

    # Rotation-invariant check: the embedding traces a fixed-radius circle
    # in the leading 2D eigenspace.
    radius = np.linalg.norm(result.eigenvectors[:, :2], axis=1)
    assert radius.std() / radius.mean() < 0.1


def test_diffusion_map_2sphere_has_degenerate_leading_triplet():
    rng = np.random.default_rng(0)
    X = sample_sphere(2, 1500, rng)
    result = diffusion_maps(X, n_components=6)

    assert result.eigenvalues[0] / result.eigenvalues[2] < 1.15  # top 3 close together
    assert result.eigenvalues[2] / result.eigenvalues[3] > 2.0  # gap after

    radius = np.linalg.norm(result.eigenvectors[:, :3], axis=1)
    assert radius.std() / radius.mean() < 0.1


def test_diffusion_map_2torus_has_four_leading_modes():
    """Product of two circles: expect ~4 leading modes (2 pairs) separated
    from the rest by a gap. Looser degeneracy tolerance than the sphere --
    the product structure doesn't factor as cleanly under an isotropic
    Gaussian kernel bandwidth."""
    rng = np.random.default_rng(0)
    X = sample_torus(2, 1500, rng)
    result = diffusion_maps(X, n_components=8)

    assert result.eigenvalues[0] / result.eigenvalues[3] < 1.3
    assert result.eigenvalues[3] / result.eigenvalues[4] > 2.0


def test_diffusion_map_outputs_shapes_and_descending_eigenvalues():
    rng = np.random.default_rng(0)
    X = sample_sphere(5, 300, rng)
    result = diffusion_maps(X, n_components=8)
    assert result.eigenvalues.shape == (8,)
    assert result.eigenvectors.shape == (300, 8)
    assert result.embedding.shape == (300, 8)
    assert np.all(np.diff(result.eigenvalues) <= 1e-12)  # descending
