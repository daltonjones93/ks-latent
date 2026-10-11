"""Shifted tilings on rectangular grids (ks_latent/da/tilings.py), used by the Kolmogorov DA."""
import numpy as np
import pytest
import torch

from ks_latent.da.jacobian_coloring import obs_site_map_2d
from ks_latent.da.tilings import blend_weights, frame_obs_sites, hann_1d, tiling_offsets

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n,axes", [(1, "xy"), (2, "xy"), (4, "xy"), (2, "x"), (4, "x")])
def test_blend_weights_partition_of_unity_on_rectangle(n, axes):
    W = blend_weights(384, 128, 16, tiling_offsets(16, n, axes))
    assert len(W) == n
    assert np.allclose(sum(W), 1.0, atol=1e-12)
    assert all((w >= 0).all() for w in W)


def test_half_patch_hann_pair_sums_to_one_unnormalized():
    s = hann_1d(64, 16, 0) + hann_1d(64, 16, 8)
    assert np.allclose(s, 1.0, atol=1e-12)


def test_hann_smallest_at_own_patch_edges():
    w = hann_1d(64, 16, 8)
    edges = (np.arange(64) - 8) % 16
    assert w[(edges == 0) | (edges == 15)].max() < w[(edges == 7) | (edges == 8)].min()


def test_x_only_offsets():
    assert tiling_offsets(16, 4, "x") == [(0, 0), (4, 0), (8, 0), (12, 0)]
    assert all(oy == 0 for _, oy in tiling_offsets(16, 2, "x"))


def test_frame_sites_match_reference_at_zero_offset():
    _, _, site = frame_obs_sites(256, 128, 16, 8, (0, 0))
    assert torch.equal(site, obs_site_map_2d(256, 128, 16, 8, 8, 8).to(torch.int64))


@pytest.mark.parametrize("o", [(8, 0), (0, 8), (8, 8), (4, 0)])
def test_frame_obs_pick_the_same_physical_points(o):
    F = np.random.default_rng(0).standard_normal((256, 128))
    ix, iy, site = frame_obs_sites(256, 128, 16, 8, o)
    R = np.roll(F, (-o[0], -o[1]), axis=(0, 1))  # tiling o's frame
    assert np.array_equal(R[ix.numpy()][:, iy.numpy()], F[::8, ::8])
    assert int(site.max()) < (256 // 16) * (128 // 16)
