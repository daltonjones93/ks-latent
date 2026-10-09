"""Tests for `KSAutoencoderLocalField2D`/`LocalFieldAutoencoderConfig2D`
(added 2026-09-28, `docs/ks_local_field_ltransfer.tex` "Staged next
steps" Step 1) -- 2D generalization of `KSAutoencoderLocalField`, same
Gate-7-style discipline (exact equivariance, gauge anchor, no-bias
output) as `tests/unit/test_autoencoder_local_field.py`.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import LocalFieldAutoencoderConfig2D
from ks_latent.models.autoencoder_local_field_2d import KSAutoencoderLocalField2D


def _cfg(**overrides):
    defaults = dict(
        NX=32, NY=32, n_sites_x=8, n_sites_y=8, local_channels=3,
        site_mix_radius=2, n_site_mix_layers=3, hidden=32,
    )
    defaults.update(overrides)
    return LocalFieldAutoencoderConfig2D(**defaults)


def test_d_latent():
    cfg = _cfg()
    assert cfg.d_latent == 8 * 8 * 3 == 192
    assert cfg.patch_x == 4
    assert cfg.patch_y == 4


def test_forward_shapes():
    cfg = _cfg()
    ae = KSAutoencoderLocalField2D(cfg)
    u = torch.randn(4, cfg.NX * cfg.NY)
    u_hat, z = ae(u)
    assert u_hat.shape == (4, cfg.NX * cfg.NY)
    assert z.shape == (4, cfg.d_latent)
    assert torch.isfinite(u_hat).all()
    assert torch.isfinite(z).all()


def test_gauge_anchor_is_exact_physical_patch_average():
    cfg = _cfg()
    ae = KSAutoencoderLocalField2D(cfg)
    u_flat = torch.randn(3, cfg.NX * cfg.NY)
    z = ae.encode(u_flat)
    u = u_flat.reshape(3, cfg.NX, cfg.NY)
    expected = u.reshape(3, cfg.n_sites_x, cfg.patch_x, cfg.n_sites_y, cfg.patch_y).mean(dim=(2, 4))
    z_field = z.reshape(3, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    assert torch.allclose(z_field[..., 0], expected, atol=1e-5)


def test_exact_translation_equivariance_x_axis():
    torch.manual_seed(0)
    cfg = _cfg()
    ae = KSAutoencoderLocalField2D(cfg).eval()
    u = torch.randn(2, cfg.NX, cfg.NY)
    with torch.no_grad():
        z = ae.encode(u.reshape(2, -1))
        z_shifted = ae.encode(torch.roll(u, shifts=cfg.patch_x, dims=1).reshape(2, -1))
    z_field = z.reshape(2, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    z_field_shifted = z_shifted.reshape(2, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    z_field_rolled = torch.roll(z_field, shifts=1, dims=1)
    assert torch.allclose(z_field_shifted, z_field_rolled, atol=1e-5)


def test_exact_translation_equivariance_y_axis():
    torch.manual_seed(0)
    cfg = _cfg()
    ae = KSAutoencoderLocalField2D(cfg).eval()
    u = torch.randn(2, cfg.NX, cfg.NY)
    with torch.no_grad():
        z = ae.encode(u.reshape(2, -1))
        z_shifted = ae.encode(torch.roll(u, shifts=cfg.patch_y, dims=2).reshape(2, -1))
    z_field = z.reshape(2, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    z_field_shifted = z_shifted.reshape(2, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    z_field_rolled = torch.roll(z_field, shifts=1, dims=2)
    assert torch.allclose(z_field_shifted, z_field_rolled, atol=1e-5)


def test_exact_translation_equivariance_both_axes():
    torch.manual_seed(0)
    cfg = _cfg()
    ae = KSAutoencoderLocalField2D(cfg).eval()
    u = torch.randn(2, cfg.NX, cfg.NY)
    shift_x_sites, shift_y_sites = 3, 2
    u_shifted = torch.roll(
        torch.roll(u, shifts=shift_x_sites * cfg.patch_x, dims=1), shifts=shift_y_sites * cfg.patch_y, dims=2
    )
    with torch.no_grad():
        z = ae.encode(u.reshape(2, -1))
        z_shifted = ae.encode(u_shifted.reshape(2, -1))
    z_field = z.reshape(2, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    z_field_shifted = z_shifted.reshape(2, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    z_field_rolled = torch.roll(
        torch.roll(z_field, shifts=shift_x_sites, dims=1), shifts=shift_y_sites, dims=2
    )
    assert torch.allclose(z_field_shifted, z_field_rolled, atol=1e-5)


def test_no_bias_on_learned_output_conv():
    cfg = _cfg()
    ae = KSAutoencoderLocalField2D(cfg)
    assert ae.enc_out.bias is None


def test_gradients_flow():
    cfg = _cfg()
    ae = KSAutoencoderLocalField2D(cfg)
    u = torch.randn(4, cfg.NX * cfg.NY, requires_grad=True)
    u_hat, z = ae(u)
    loss = u_hat.pow(2).sum() + z.pow(2).sum()
    loss.backward()
    assert ae.enc_mix[0].weight.grad is not None and ae.enc_mix[0].weight.grad.abs().sum() > 0
    assert u.grad is not None and u.grad.abs().sum() > 0


def test_invalid_nx_rejected():
    with pytest.raises(ValueError, match="NX"):
        _cfg(NX=30, n_sites_x=8)  # 30 not divisible by 8


def test_decode_jacobian_is_exactly_block_diagonal_at_zero_reach():
    """DA motivation (docs/ks_local_field_ltransfer.tex sec:consolidation
    Phase 4, user-directed 2026-10-06): a decoder with reach 0 (encoder
    reach is irrelevant here -- only the DECODER is in the observation
    operator's forward path) should give every physical pixel a nonzero
    gradient w.r.t. EXACTLY one site's latent channels, and an EXACTLY
    zero gradient w.r.t. every other site -- the property that makes the
    observation Jacobian in 2D latent DA (`ks_latent/da/letkf.py`,
    `ks_latent/da/pff.py`) genuinely block-diagonal/sparse, not merely
    small in magnitude far away."""
    import torch
    from torch.func import jacrev
    from ks_latent.config import LocalFieldAutoencoderConfig2D
    from ks_latent.models.autoencoder_local_field_2d import KSAutoencoderLocalField2D

    cfg = LocalFieldAutoencoderConfig2D(
        NX=64, NY=64, n_sites_x=4, n_sites_y=4, local_channels=3,
        site_mix_radius=2, n_site_mix_layers=3,  # encoder reach nonzero: must not matter here
        dec_site_mix_radius=0, dec_n_site_mix_layers=0,
    )
    torch.manual_seed(0)
    ae = KSAutoencoderLocalField2D(cfg).double().eval()
    d = cfg.n_sites_x * cfg.n_sites_y * cfg.local_channels
    z = torch.randn(d, dtype=torch.float64)

    def decode_one(z_single):
        return ae.decode(z_single.unsqueeze(0)).squeeze(0)

    J = jacrev(decode_one)(z).reshape(cfg.NX, cfg.NY, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    px, py = cfg.patch_x, cfg.patch_y
    for site_x in range(cfg.n_sites_x):
        for site_y in range(cfg.n_sites_y):
            gx, gy = site_x * px + px // 2, site_y * py + py // 2  # a pixel inside this site's patch
            grad_mag = J[gx, gy].abs().sum(dim=-1)  # (n_sites_x, n_sites_y)
            on_site = grad_mag[site_x, site_y].item()
            off_site_max = grad_mag.clone()
            off_site_max[site_x, site_y] = 0.0
            assert on_site > 1e-10, f"site ({site_x},{site_y}) has zero self-gradient"
            assert off_site_max.max().item() == 0.0, (
                f"site ({site_x},{site_y}): nonzero gradient to another site "
                f"(max {off_site_max.max().item()}) -- decoder reach is not actually 0"
            )
