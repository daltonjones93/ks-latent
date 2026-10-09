"""Tests for `KSAutoencoderLocalAttention2D`/`LocalAttentionAutoencoderConfig2D`
(`encoder_kind="local_attention_2d"`, added 2026-09-29, user-directed:
"please implement the local attention variant you described above" --
`CLAUDE_CODE_BRIEF.md` Sec. 12.2.2). Mirrors
`test_autoencoder_local_field_2d.py`'s discipline (exact equivariance,
gauge anchor, no-bias output) plus the brief's own explicit windowed-
attention tests: mask correctness, shared (not per-patch) relative bias,
and a receptive-field UPPER BOUND (not equality -- attention may
concentrate more narrowly than the analytic `n_layers*window` bound).
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import LocalAttentionAutoencoderConfig2D
from ks_latent.models.autoencoder_local_attention_2d import (
    KSAutoencoderLocalAttention2D,
    _WindowedAttnBlock2D,
)


def _cfg(**overrides):
    defaults = dict(
        NX=32, NY=32, n_sites_x=8, n_sites_y=8, local_channels=3,
        window=2, n_site_mix_layers=2, d_model=16, n_heads=2, mlp_ratio=2,
    )
    defaults.update(overrides)
    return LocalAttentionAutoencoderConfig2D(**defaults)


def test_d_latent():
    cfg = _cfg()
    assert cfg.d_latent == 8 * 8 * 3 == 192
    assert cfg.patch_x == 4
    assert cfg.patch_y == 4


def test_forward_shapes():
    cfg = _cfg()
    ae = KSAutoencoderLocalAttention2D(cfg)
    u = torch.randn(4, cfg.NX * cfg.NY)
    u_hat, z = ae(u)
    assert u_hat.shape == (4, cfg.NX * cfg.NY)
    assert z.shape == (4, cfg.d_latent)
    assert torch.isfinite(u_hat).all()
    assert torch.isfinite(z).all()


def test_gauge_anchor_is_exact_physical_patch_average():
    cfg = _cfg()
    ae = KSAutoencoderLocalAttention2D(cfg)
    u_flat = torch.randn(3, cfg.NX * cfg.NY)
    z = ae.encode(u_flat)
    u = u_flat.reshape(3, cfg.NX, cfg.NY)
    expected = u.reshape(3, cfg.n_sites_x, cfg.patch_x, cfg.n_sites_y, cfg.patch_y).mean(dim=(2, 4))
    z_field = z.reshape(3, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    assert torch.allclose(z_field[..., 0], expected, atol=1e-5)


def test_exact_translation_equivariance_x_axis():
    torch.manual_seed(0)
    cfg = _cfg()
    ae = KSAutoencoderLocalAttention2D(cfg).eval()
    u = torch.randn(2, cfg.NX, cfg.NY)
    with torch.no_grad():
        z = ae.encode(u.reshape(2, -1))
        z_shifted = ae.encode(torch.roll(u, shifts=cfg.patch_x, dims=1).reshape(2, -1))
    z_field = z.reshape(2, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    z_field_shifted = z_shifted.reshape(2, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    z_field_rolled = torch.roll(z_field, shifts=1, dims=1)
    assert torch.allclose(z_field_shifted, z_field_rolled, atol=1e-4)


def test_exact_translation_equivariance_y_axis():
    torch.manual_seed(0)
    cfg = _cfg()
    ae = KSAutoencoderLocalAttention2D(cfg).eval()
    u = torch.randn(2, cfg.NX, cfg.NY)
    with torch.no_grad():
        z = ae.encode(u.reshape(2, -1))
        z_shifted = ae.encode(torch.roll(u, shifts=cfg.patch_y, dims=2).reshape(2, -1))
    z_field = z.reshape(2, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    z_field_shifted = z_shifted.reshape(2, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
    z_field_rolled = torch.roll(z_field, shifts=1, dims=2)
    assert torch.allclose(z_field_shifted, z_field_rolled, atol=1e-4)


def test_exact_translation_equivariance_both_axes():
    torch.manual_seed(0)
    cfg = _cfg()
    ae = KSAutoencoderLocalAttention2D(cfg).eval()
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
    assert torch.allclose(z_field_shifted, z_field_rolled, atol=1e-4)


def test_no_bias_on_learned_output_projection():
    cfg = _cfg()
    ae = KSAutoencoderLocalAttention2D(cfg)
    assert ae.enc_out.bias is None


def test_gradients_flow():
    cfg = _cfg()
    ae = KSAutoencoderLocalAttention2D(cfg)
    u = torch.randn(4, cfg.NX * cfg.NY, requires_grad=True)
    u_hat, z = ae(u)
    loss = u_hat.pow(2).sum() + z.pow(2).sum()
    loss.backward()
    assert ae.enc_blocks[0].qkv.weight.grad is not None and ae.enc_blocks[0].qkv.weight.grad.abs().sum() > 0
    assert ae.enc_blocks[0].bias_table.grad is not None and ae.enc_blocks[0].bias_table.grad.abs().sum() > 0
    assert u.grad is not None and u.grad.abs().sum() > 0


def test_invalid_nx_rejected():
    with pytest.raises(ValueError, match="NX"):
        _cfg(NX=30, n_sites_x=8)  # 30 not divisible by 8


def test_d_model_not_divisible_by_n_heads_rejected():
    with pytest.raises(ValueError, match="d_model"):
        _cfg(d_model=15, n_heads=2)


# --- windowed-attention-specific tests (CLAUDE_CODE_BRIEF.md Sec. 12.2.2) ---


def test_windowed_attention_mask_is_circular_banded():
    """Mask True iff per-axis circular distance <= window on BOTH axes,
    including wraparound at the P-1/0 boundary."""
    n_sites_x, n_sites_y, window = 6, 6, 1
    block = _WindowedAttnBlock2D(
        d_model=8, n_heads=2, window=window, mlp_ratio=2, dropout=0.0,
        n_sites_x=n_sites_x, n_sites_y=n_sites_y,
    )
    mask = block.mask.reshape(n_sites_x, n_sites_y, n_sites_x, n_sites_y)
    for ix in range(n_sites_x):
        for iy in range(n_sites_y):
            for jx in range(n_sites_x):
                for jy in range(n_sites_y):
                    dx = min(abs(ix - jx), n_sites_x - abs(ix - jx))
                    dy = min(abs(iy - jy), n_sites_y - abs(iy - jy))
                    expected = (dx <= window) and (dy <= window)
                    assert bool(mask[ix, iy, jx, jy]) == expected, (ix, iy, jx, jy)
    # Explicit wraparound check: site 0 and site (P-1) are circular neighbors.
    assert bool(mask[0, 0, n_sites_x - 1, 0])


def test_relative_position_bias_shared_not_per_patch():
    """The bias table's shape depends only on (n_heads, window+1, window+1)
    -- NOT on n_sites_x/n_sites_y -- i.e. no per-patch-index parameters."""
    block_small = _WindowedAttnBlock2D(
        d_model=8, n_heads=2, window=2, mlp_ratio=2, dropout=0.0, n_sites_x=6, n_sites_y=6,
    )
    block_large = _WindowedAttnBlock2D(
        d_model=8, n_heads=2, window=2, mlp_ratio=2, dropout=0.0, n_sites_x=20, n_sites_y=20,
    )
    assert block_small.bias_table.shape == block_large.bias_table.shape == (2, 3, 3)


def test_structural_buffers_not_in_state_dict():
    """mask/dx_idx/dy_idx must be persistent=False -- otherwise resizing
    to a different n_sites_x/n_sites_y could never load_state_dict(strict=True)."""
    block = _WindowedAttnBlock2D(
        d_model=8, n_heads=2, window=1, mlp_ratio=2, dropout=0.0, n_sites_x=6, n_sites_y=6,
    )
    sd_keys = set(block.state_dict().keys())
    assert "mask" not in sd_keys
    assert "dx_idx" not in sd_keys
    assert "dy_idx" not in sd_keys


def test_receptive_field_bound():
    """Perturbing a single input patch must leave decoded-latent sites
    farther than n_site_mix_layers*window (circular Chebyshev distance)
    EXACTLY unaffected (gradient == 0) -- an upper bound guaranteed by
    the mask, per CLAUDE_CODE_BRIEF.md's own caveat that the learned
    attention may concentrate more narrowly than this bound."""
    torch.manual_seed(0)
    n_sites = 9  # odd, so there's a clear "far side" with room on all sides
    window = 1
    n_layers = 2
    cfg = _cfg(
        NX=n_sites * 2, NY=n_sites * 2, n_sites_x=n_sites, n_sites_y=n_sites,
        window=window, n_site_mix_layers=n_layers, d_model=8, n_heads=2, mlp_ratio=2,
    )
    ae = KSAutoencoderLocalAttention2D(cfg).eval()
    u = torch.randn(1, cfg.NX * cfg.NY, requires_grad=True)
    z = ae.encode(u)
    z_field = z.reshape(1, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)

    query_site = (0, 0)
    max_reach = n_layers * window
    # Farthest reachable site (residual/attention channel, index 1, not the
    # gauge anchor -- the anchor is a fixed physical average local to its
    # OWN patch only, by construction, not run through any attention layer).
    target = z_field[0, query_site[0], query_site[1], 1]
    grad = torch.autograd.grad(target, u, retain_graph=False)[0]
    grad_by_site = grad.reshape(cfg.NX, cfg.NY).reshape(
        cfg.n_sites_x, cfg.patch_x, cfg.n_sites_y, cfg.patch_y
    ).abs().sum(dim=(1, 3))  # (n_sites_x, n_sites_y)

    for ix in range(n_sites):
        for iy in range(n_sites):
            dx = min(abs(ix - query_site[0]), n_sites - abs(ix - query_site[0]))
            dy = min(abs(iy - query_site[1]), n_sites - abs(iy - query_site[1]))
            if max(dx, dy) > max_reach:
                assert grad_by_site[ix, iy].item() == 0.0, (
                    f"site ({ix},{iy}) at distance {max(dx, dy)} > bound {max_reach} "
                    f"received nonzero gradient {grad_by_site[ix, iy].item()}"
                )
