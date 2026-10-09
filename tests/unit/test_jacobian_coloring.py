"""Tests for `ks_latent.da.jacobian_coloring` (T1,
docs/ks_local_field_ltransfer.tex sec:c19_da_hypotheses). Ground rule 1:
every numerical claim gets a test against a known answer -- here, the
already-validated `jacrev` path (`ks_latent.da.letkf._jacobian_at`) is
the ground truth the coloured Jacobian must reproduce exactly, plus a
direct test that it costs a FIXED number of function evaluations
regardless of how many observations there are (H3's "flat cost" claim).
"""

from __future__ import annotations

import pytest
import torch
from torch.func import jacrev

from ks_latent.config import LocalFieldAutoencoderConfig2D
from ks_latent.da.jacobian_coloring import (
    assert_decoder_zero_reach,
    coloured_obs_jacobian_2d,
    coloured_obs_jacobian_2d_compact,
    obs_site_map_2d,
)
from ks_latent.models.autoencoder_local_field_2d import KSAutoencoderLocalField2D


def _zero_reach_ae(n_sites_x=4, n_sites_y=4, patch=4, channels=3, seed=0):
    NX = n_sites_x * patch
    NY = n_sites_y * patch
    cfg = LocalFieldAutoencoderConfig2D(
        NX=NX, NY=NY, n_sites_x=n_sites_x, n_sites_y=n_sites_y, local_channels=channels,
        site_mix_radius=2, n_site_mix_layers=2,  # encoder reach nonzero: must not matter
        dec_site_mix_radius=0, dec_n_site_mix_layers=0,
    )
    torch.manual_seed(seed)
    ae = KSAutoencoderLocalField2D(cfg).double().eval()
    return ae, cfg


def _sparse_obs_operator_2d(NX, NY, stride_x, stride_y):
    def op(u_phys):
        return u_phys.reshape(NX, NY)[::stride_x, ::stride_y].reshape(-1)
    return op


# --- obs_site_map_2d --------------------------------------------------------

def test_obs_site_map_2d_hand_computed_every_pixel():
    # NX=NY=4, n_sites=2x2, patch=2x2: ix,iy in 0..3, stride 1 (every pixel).
    site = obs_site_map_2d(NX=4, NY=4, n_sites_x=2, n_sites_y=2, stride_x=1, stride_y=1)
    expected = torch.tensor([
        0, 0, 1, 1,
        0, 0, 1, 1,
        2, 2, 3, 3,
        2, 2, 3, 3,
    ], dtype=torch.int64)
    assert torch.equal(site, expected)


def test_obs_site_map_2d_one_obs_per_site():
    site = obs_site_map_2d(NX=4, NY=4, n_sites_x=2, n_sites_y=2, stride_x=2, stride_y=2)
    assert torch.equal(site, torch.tensor([0, 1, 2, 3], dtype=torch.int64))


def test_obs_site_map_2d_raises_on_indivisible_size():
    with pytest.raises(ValueError):
        obs_site_map_2d(NX=5, NY=4, n_sites_x=2, n_sites_y=2, stride_x=1, stride_y=1)


# --- assert_decoder_zero_reach ----------------------------------------------

def test_assert_decoder_zero_reach_passes_for_zero_radius():
    _, cfg = _zero_reach_ae()
    assert_decoder_zero_reach(cfg)  # must not raise


def test_assert_decoder_zero_reach_passes_for_zero_layers():
    cfg = LocalFieldAutoencoderConfig2D(
        NX=16, NY=16, n_sites_x=4, n_sites_y=4, local_channels=3,
        site_mix_radius=0, n_site_mix_layers=0,
        dec_site_mix_radius=3, dec_n_site_mix_layers=0,  # radius nonzero but zero LAYERS: still zero reach
    )
    assert_decoder_zero_reach(cfg)  # must not raise


def test_assert_decoder_zero_reach_raises_for_nonzero_reach():
    cfg = LocalFieldAutoencoderConfig2D(
        NX=16, NY=16, n_sites_x=4, n_sites_y=4, local_channels=3,
        site_mix_radius=0, n_site_mix_layers=0,
        dec_site_mix_radius=1, dec_n_site_mix_layers=2,
    )
    with pytest.raises(ValueError, match="decoder reach 0"):
        assert_decoder_zero_reach(cfg)


# --- coloured_obs_jacobian_2d vs jacrev --------------------------------------

@pytest.mark.parametrize("stride", [2, 4])
def test_coloured_jacobian_matches_jacrev_exactly(stride):
    ae, cfg = _zero_reach_ae(n_sites_x=4, n_sites_y=4, patch=4, channels=3)
    assert_decoder_zero_reach(cfg)  # precondition holds
    NX, NY = cfg.NX, cfg.NY
    d = cfg.n_sites_x * cfg.n_sites_y * cfg.local_channels
    obs_op = _sparse_obs_operator_2d(NX, NY, stride, stride)

    def h(z_single):
        u_phys = ae.decode(z_single.unsqueeze(0)).squeeze(0)
        return obs_op(u_phys)

    torch.manual_seed(1)
    z_bar = torch.randn(d, dtype=torch.float64)

    J_true = jacrev(h)(z_bar)  # (m, d), ground truth
    site_of_obs = obs_site_map_2d(NX, NY, cfg.n_sites_x, cfg.n_sites_y, stride, stride)
    J_coloured = coloured_obs_jacobian_2d(h, z_bar, site_of_obs, cfg.local_channels)

    assert J_coloured.shape == J_true.shape
    max_err = (J_coloured - J_true).abs().max().item()
    assert max_err < 1e-9, f"coloured Jacobian disagrees with jacrev by {max_err}"


def test_coloured_jacobian_compact_matches_dense_gather():
    ae, cfg = _zero_reach_ae(n_sites_x=3, n_sites_y=3, patch=2, channels=4)
    NX, NY = cfg.NX, cfg.NY
    d = cfg.n_sites_x * cfg.n_sites_y * cfg.local_channels
    obs_op = _sparse_obs_operator_2d(NX, NY, 2, 2)

    def h(z_single):
        return obs_op(ae.decode(z_single.unsqueeze(0)).squeeze(0))

    torch.manual_seed(2)
    z_bar = torch.randn(d, dtype=torch.float64)
    site_of_obs = obs_site_map_2d(NX, NY, cfg.n_sites_x, cfg.n_sites_y, 2, 2)

    J_compact = coloured_obs_jacobian_2d_compact(h, z_bar, site_of_obs, cfg.local_channels)
    J_dense = coloured_obs_jacobian_2d(h, z_bar, site_of_obs, cfg.local_channels)

    m = site_of_obs.shape[0]
    for i in range(m):
        s = site_of_obs[i].item()
        gathered = J_dense[i, s * cfg.local_channels:(s + 1) * cfg.local_channels]
        assert torch.allclose(gathered, J_compact[i])
        # everything outside this site's block must be exactly zero.
        mask = torch.ones(d, dtype=torch.bool)
        mask[s * cfg.local_channels:(s + 1) * cfg.local_channels] = False
        assert J_dense[i, mask].abs().max().item() == 0.0


def test_coloured_jacobian_cost_is_flat_in_number_of_observations():
    """H3's claim: cost (number of calls to `h`) is `channels`,
    independent of `m`. Direct test via a call-counting wrapper, at two
    very different observation densities."""
    ae, cfg = _zero_reach_ae(n_sites_x=6, n_sites_y=6, patch=2, channels=3)
    NX, NY = cfg.NX, cfg.NY
    d = cfg.n_sites_x * cfg.n_sites_y * cfg.local_channels
    torch.manual_seed(3)
    z_bar = torch.randn(d, dtype=torch.float64)

    counts = {}
    for stride, label in [(2, "dense_obs"), (6, "sparse_obs")]:
        obs_op = _sparse_obs_operator_2d(NX, NY, stride, stride)
        call_count = [0]

        def h(z_single, obs_op=obs_op, call_count=call_count):
            call_count[0] += 1
            return obs_op(ae.decode(z_single.unsqueeze(0)).squeeze(0))

        site_of_obs = obs_site_map_2d(NX, NY, cfg.n_sites_x, cfg.n_sites_y, stride, stride)
        coloured_obs_jacobian_2d_compact(h, z_bar, site_of_obs, cfg.local_channels)
        counts[label] = call_count[0]

    assert counts["dense_obs"] == cfg.local_channels
    assert counts["sparse_obs"] == cfg.local_channels
    assert counts["dense_obs"] == counts["sparse_obs"], (
        f"call count should not depend on observation density: {counts}"
    )
