"""Tests for `KSAutoencoderMultiscaleLocalField`/
`MultiscaleLocalFieldAutoencoderConfig` (added 2026-09-28, user-directed:
combine Section 224's local_field/local_mlp recipe with the multiscale
latent idea in docs/spatially_structured_latent_DA_FMM_research_idea.md).
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import MultiscaleLocalFieldAutoencoderConfig
from ks_latent.models.autoencoder_multiscale_local_field import KSAutoencoderMultiscaleLocalField


def _cfg(**overrides):
    defaults = dict(
        NX=256, fine_n_sites=16, fine_local_channels=3, fine_site_mix_radius=2,
        fine_n_site_mix_layers=3, hidden=32, regional_pool_factor=4, regional_local_channels=3,
        global_pool_factor=4, global_local_channels=2,
    )
    defaults.update(overrides)
    return MultiscaleLocalFieldAutoencoderConfig(**defaults)


def test_d_latent_matches_sum_of_blocks():
    cfg = _cfg()
    assert cfg.d_latent == 16 * 3 + 4 * 3 + 1 * 2 == 62
    assert cfg.regional_n_sites == 4
    assert cfg.global_n_sites == 1


def test_forward_shapes():
    cfg = _cfg()
    ae = KSAutoencoderMultiscaleLocalField(cfg)
    u = torch.randn(4, cfg.NX)
    u_hat, z = ae(u)
    assert u_hat.shape == (4, cfg.NX)
    assert z.shape == (4, cfg.d_latent)
    assert torch.isfinite(u_hat).all()
    assert torch.isfinite(z).all()


def test_fine_gauge_anchor_is_exact_physical_patch_average():
    cfg = _cfg()
    ae = KSAutoencoderMultiscaleLocalField(cfg)
    u = torch.randn(3, cfg.NX)
    z = ae.encode(u)
    patch = cfg.NX // cfg.fine_n_sites
    expected = u.reshape(3, cfg.fine_n_sites, patch).mean(dim=-1)
    z_fine = z[:, : cfg.fine_n_sites * cfg.fine_local_channels].reshape(
        3, cfg.fine_n_sites, cfg.fine_local_channels
    )
    assert torch.allclose(z_fine[:, :, 0], expected, atol=1e-5)


def test_regional_gauge_anchor_is_exact_physical_patch_average():
    cfg = _cfg()
    ae = KSAutoencoderMultiscaleLocalField(cfg)
    u = torch.randn(3, cfg.NX)
    z = ae.encode(u)
    regional_patch = (cfg.NX // cfg.fine_n_sites) * cfg.regional_pool_factor
    expected = u.reshape(3, cfg.regional_n_sites, regional_patch).mean(dim=-1)
    d_fine = cfg.fine_n_sites * cfg.fine_local_channels
    d_regional = cfg.regional_n_sites * cfg.regional_local_channels
    z_regional = z[:, d_fine : d_fine + d_regional].reshape(3, cfg.regional_n_sites, cfg.regional_local_channels)
    assert torch.allclose(z_regional[:, :, 0], expected, atol=1e-5)


def test_global_gauge_anchor_equals_conserved_mean_of_u():
    """For KS's periodic BC, mean(u) is an exactly conserved quantity --
    the global scale's channel-0 anchor is this quantity directly."""
    cfg = _cfg()
    ae = KSAutoencoderMultiscaleLocalField(cfg)
    u = torch.randn(5, cfg.NX)
    z = ae.encode(u)
    d_fine = cfg.fine_n_sites * cfg.fine_local_channels
    d_regional = cfg.regional_n_sites * cfg.regional_local_channels
    global_anchor = z[:, d_fine + d_regional]
    assert torch.allclose(global_anchor, u.mean(dim=1), atol=1e-5)


def test_fine_scale_exact_translation_equivariance():
    torch.manual_seed(0)
    cfg = _cfg()
    ae = KSAutoencoderMultiscaleLocalField(cfg).eval()
    u = torch.randn(2, cfg.NX)
    patch = cfg.NX // cfg.fine_n_sites
    with torch.no_grad():
        z = ae.encode(u)
        z_shifted = ae.encode(torch.roll(u, shifts=patch, dims=1))
    d_fine = cfg.fine_n_sites * cfg.fine_local_channels
    z_fine = z[:, :d_fine].reshape(2, cfg.fine_n_sites, cfg.fine_local_channels)
    z_fine_shifted = z_shifted[:, :d_fine].reshape(2, cfg.fine_n_sites, cfg.fine_local_channels)
    z_fine_rolled = torch.roll(z_fine, shifts=1, dims=1)
    assert torch.allclose(z_fine_shifted, z_fine_rolled, atol=1e-5)


def test_regional_scale_exact_equivariance_under_regional_patch_shift():
    torch.manual_seed(0)
    cfg = _cfg()
    ae = KSAutoencoderMultiscaleLocalField(cfg).eval()
    u = torch.randn(2, cfg.NX)
    patch = cfg.NX // cfg.fine_n_sites
    shift = patch * cfg.regional_pool_factor  # exactly one regional patch
    with torch.no_grad():
        z = ae.encode(u)
        z_shifted = ae.encode(torch.roll(u, shifts=shift, dims=1))
    d_fine = cfg.fine_n_sites * cfg.fine_local_channels
    d_regional = cfg.regional_n_sites * cfg.regional_local_channels
    z_regional = z[:, d_fine : d_fine + d_regional].reshape(2, cfg.regional_n_sites, cfg.regional_local_channels)
    z_regional_shifted = z_shifted[:, d_fine : d_fine + d_regional].reshape(
        2, cfg.regional_n_sites, cfg.regional_local_channels
    )
    z_regional_rolled = torch.roll(z_regional, shifts=1, dims=1)
    assert torch.allclose(z_regional_shifted, z_regional_rolled, atol=1e-5)


def test_global_anchor_is_shift_invariant():
    torch.manual_seed(0)
    cfg = _cfg()
    ae = KSAutoencoderMultiscaleLocalField(cfg).eval()
    u = torch.randn(2, cfg.NX)
    with torch.no_grad():
        z = ae.encode(u)
        z_shifted = ae.encode(torch.roll(u, shifts=7, dims=1))  # arbitrary, non-aligned shift
    d_fine = cfg.fine_n_sites * cfg.fine_local_channels
    d_regional = cfg.regional_n_sites * cfg.regional_local_channels
    global_anchor = z[:, d_fine + d_regional]
    global_anchor_shifted = z_shifted[:, d_fine + d_regional]
    assert torch.allclose(global_anchor, global_anchor_shifted, atol=1e-4)


def test_no_bias_on_learned_output_convs():
    """Matches KSAutoencoderLocalField's own fix (Section 153): a
    learnable per-channel bias on the residual-channel output conv lets
    it drift to an arbitrary constant, aliasing onto a self-FFT mode."""
    cfg = _cfg()
    ae = KSAutoencoderMultiscaleLocalField(cfg)
    assert ae.enc_out_fine.bias is None
    assert ae.enc_out_regional.bias is None
    assert ae.enc_out_global.bias is None


def test_gradients_flow_to_all_scales():
    cfg = _cfg()
    ae = KSAutoencoderMultiscaleLocalField(cfg)
    u = torch.randn(4, cfg.NX, requires_grad=True)
    u_hat, z = ae(u)
    loss = u_hat.pow(2).sum() + z.pow(2).sum()
    loss.backward()
    assert ae.enc_mix[0].weight.grad is not None and ae.enc_mix[0].weight.grad.abs().sum() > 0
    assert ae.regional_mix.weight.grad is not None and ae.regional_mix.weight.grad.abs().sum() > 0
    assert ae.enc_out_global.weight.grad is not None and ae.enc_out_global.weight.grad.abs().sum() > 0
    assert u.grad is not None and u.grad.abs().sum() > 0


def test_invalid_global_pool_factor_rejected():
    with pytest.raises(ValueError, match="global_n_sites"):
        _cfg(global_pool_factor=2)  # regional_n_sites=4, /2 = 2 != 1


def test_invalid_regional_pool_factor_rejected():
    with pytest.raises(ValueError, match="regional_pool_factor"):
        _cfg(fine_n_sites=16, regional_pool_factor=3)  # 16 not divisible by 3
