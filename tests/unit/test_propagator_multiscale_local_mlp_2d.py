"""Tests for `backbone="multiscale_local_mlp_2d"`
(`_MultiscaleLocalMLPDeltaBody2D`, added 2026-10-01, D3 of
docs/oct_1_exp.md -- user-directed: "please make very sure to design
trainable dynamic interactions between the localized dynamics coming from
the fixed 'patched' model"). Mirrors the other local_mlp_2d* test files'
discipline; the equivariance tests are the key new piece (exact at
regional-block-multiple shifts, deliberately NOT exact at generic site
shifts -- ground rule 19, same precedent as `local_mlp_2d_aniso`'s
y-phase and the encoder's own patchify)."""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import AuxPropagatorConfig, PropagatorConfig
from ks_latent.models import load_propagator_checkpoint_resized_2d_multiscale
from ks_latent.models.propagator import AuxPropagator, LatentPropagator


def _cfg(cls=PropagatorConfig, **overrides):
    defaults = dict(
        d_latent=192, mode="markovian", backbone="multiscale_local_mlp_2d", attn_window=2,
        local_mlp2d_n_sites_x=8, local_mlp2d_n_sites_y=8,
        local_mlp2d_local_channels=3, local_mlp2d_token_d_model=32,
        local_mlp2d_n_layers=2, multiscale_pool_x=2, multiscale_pool_y=2,
        multiscale_regional_token_d_model=16, multiscale_regional_n_layers=2,
        multiscale_regional_window=1,
    )
    defaults.update(overrides)
    return cls(**defaults)


def test_shape_and_finite():
    cfg = _cfg()
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    assert out.shape == (4, 192)
    assert torch.isfinite(out).all()


def test_identity_at_init():
    cfg = _cfg(zero_init=True)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    assert torch.allclose(out, z, atol=1e-5)


def test_not_identity_when_zero_init_false():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    assert not torch.allclose(out, z, atol=1e-5)


def test_gradient_flows_through_both_scales():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    out.sum().backward()
    body = prop.body
    for name, p in [
        ("token_embed", body.token_embed.weight),
        ("pool_project", body.pool_project.weight),
        ("regional block mix", body.regional_blocks[0].mix.weight),
        ("broadcast_project", body.broadcast_project.weight),
        ("fine block mix", body.fine_blocks[0].mix.weight),
    ]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} has no gradient"


def test_exact_equivariance_under_regional_block_shift():
    """A shift by a whole multiple of (pool_x, pool_y) sites keeps
    pooling-block boundaries aligned, so this must be EXACT."""
    torch.manual_seed(0)
    cfg = _cfg(zero_init=False, dropout=0.0)
    prop = LatentPropagator(cfg).eval()
    Px, Py, c = 8, 8, 3
    pool_x, pool_y = 2, 2
    z = torch.randn(2, Px * Py * c)
    z_field = z.reshape(2, Px, Py, c)
    z_shifted = torch.roll(z_field, shifts=(pool_x, pool_y), dims=(1, 2)).reshape(2, Px * Py * c)
    with torch.no_grad():
        out = prop.step_one(z)
        out_shifted = prop.step_one(z_shifted)
    out_field = out.reshape(2, Px, Py, c)
    out_shifted_field = out_shifted.reshape(2, Px, Py, c)
    rolled = torch.roll(out_field, shifts=(pool_x, pool_y), dims=(1, 2))
    assert torch.allclose(out_shifted_field, rolled, atol=1e-4)


def test_generic_site_shift_not_exactly_equivariant():
    """Ground rule 19: a shift by 1 site (not a multiple of pool_x/pool_y)
    realigns which fine sites fall in which regional pooling block, so
    this must NOT be exact -- if it started passing, the regional branch
    silently stopped depending on pooling alignment (a regression)."""
    torch.manual_seed(1)
    cfg = _cfg(zero_init=False, dropout=0.0)
    prop = LatentPropagator(cfg).eval()
    Px, Py, c = 8, 8, 3
    z = torch.randn(2, Px * Py * c)
    z_field = z.reshape(2, Px, Py, c)
    z_shifted = torch.roll(z_field, shifts=1, dims=1).reshape(2, Px * Py * c)
    with torch.no_grad():
        out = prop.step_one(z)
        out_shifted = prop.step_one(z_shifted)
    out_field = out.reshape(2, Px, Py, c)
    rolled = torch.roll(out_field, shifts=1, dims=1)
    assert not torch.allclose(out_shifted.reshape(2, Px, Py, c), rolled, atol=1e-3)


def test_aux_propagator_config_works_too():
    cfg = _cfg(cls=AuxPropagatorConfig, zero_init=False)
    prop = AuxPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    assert out.shape == (4, 192)
    assert torch.isfinite(out).all()


def test_missing_attn_window_rejected():
    with pytest.raises(ValueError, match="multiscale_local_mlp_2d"):
        _cfg(attn_window=None)


def test_wrong_d_latent_rejected():
    with pytest.raises(ValueError, match="multiscale_local_mlp_2d"):
        _cfg(d_latent=99)


def test_pool_must_divide_n_sites():
    with pytest.raises(ValueError, match="multiscale_pool_x"):
        _cfg(local_mlp2d_n_sites_x=9, d_latent=9 * 8 * 3)


def test_rollout_multistep_finite():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z0 = torch.randn(4, 192)
    roll = prop.rollout(z0, z0, k=5)
    assert roll.shape == (4, 5, 192)
    assert torch.isfinite(roll).all()


# --- resize loader ---

def _make_checkpoint(tmp_path, n_sites_x=8, n_sites_y=8):
    cfg = _cfg(local_mlp2d_n_sites_x=n_sites_x, local_mlp2d_n_sites_y=n_sites_y,
               d_latent=n_sites_x * n_sites_y * 3, zero_init=False)
    prop = LatentPropagator(cfg)
    path = tmp_path / "ckpt.pt"
    torch.save({"prop_state_dict": prop.state_dict(), "prop_config": cfg}, path)
    return path


def test_resize_grows_regional_grid(tmp_path):
    path = _make_checkpoint(tmp_path, n_sites_x=8, n_sites_y=8)
    prop, new_cfg, _ = load_propagator_checkpoint_resized_2d_multiscale(
        str(path), new_n_sites_x=16, new_n_sites_y=8,
    )
    assert new_cfg.d_latent == 16 * 8 * 3
    assert prop.body.regional_nx == 8  # 16 // pool_x(2)
    assert prop.body.regional_ny == 4  # 8 // pool_y(2)
    z = torch.randn(2, new_cfg.d_latent)
    out = prop.step_one(z)
    assert out.shape == (2, new_cfg.d_latent)
    assert torch.isfinite(out).all()


def test_resize_rejects_wrong_backbone(tmp_path):
    cfg = PropagatorConfig(
        d_latent=192, mode="markovian", backbone="local_mlp_2d", attn_window=2,
        local_mlp2d_n_sites_x=8, local_mlp2d_n_sites_y=8, local_mlp2d_local_channels=3,
        local_mlp2d_token_d_model=32, local_mlp2d_n_layers=2,
    )
    prop = LatentPropagator(cfg)
    path = tmp_path / "ckpt_wrong.pt"
    torch.save({"prop_state_dict": prop.state_dict(), "prop_config": cfg}, path)
    with pytest.raises(ValueError, match="multiscale_local_mlp_2d"):
        load_propagator_checkpoint_resized_2d_multiscale(str(path), new_n_sites_x=16, new_n_sites_y=8)


def test_resize_rejects_pool_mismatch(tmp_path):
    path = _make_checkpoint(tmp_path, n_sites_x=8, n_sites_y=8)
    with pytest.raises(ValueError, match="multiscale_pool_x"):
        load_propagator_checkpoint_resized_2d_multiscale(str(path), new_n_sites_x=9, new_n_sites_y=8)
