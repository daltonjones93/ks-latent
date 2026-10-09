"""Tests for `load_autoencoder_checkpoint_resized_2d_aniso_x`/
`load_propagator_checkpoint_resized_2d_aniso_x` (`ks_latent/models/
__init__.py`, added 2026-09-29, user-directed "sure, build it" for the
Kolmogorov-flow-specific anisotropic architecture). X-ONLY resize:
`n_sites_y`/`NY` must stay fixed since the y-phase buffers are tied to
the trained `forcing_n`/`n_sites_y`, unlike the isotropic 2D case's own
resize helper (`test_ltransfer_resized_loading_2d.py`), which resizes
both axes freely.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import AuxPropagatorConfig, LocalFieldAutoencoderAniso2DConfig
from ks_latent.models import (
    load_autoencoder_checkpoint_resized_2d_aniso_x,
    load_propagator_checkpoint_resized_2d_aniso_x,
)
from ks_latent.models.autoencoder_local_field_2d_aniso import KSAutoencoderLocalFieldAniso2D
from ks_latent.models.propagator import AuxPropagator


def _save_local_field_2d_aniso_checkpoint(
    path, n_sites_x=4, n_sites_y=8, NX=32, NY=64, local_channels=2,
    site_mix_radius=1, n_site_mix_layers=2, hidden=6, forcing_n=4,
):
    cfg = LocalFieldAutoencoderAniso2DConfig(
        NX=NX, NY=NY, n_sites_x=n_sites_x, n_sites_y=n_sites_y, local_channels=local_channels,
        site_mix_radius=site_mix_radius, n_site_mix_layers=n_site_mix_layers, hidden=hidden,
        forcing_n=forcing_n,
    )
    ae = KSAutoencoderLocalFieldAniso2D(cfg)
    torch.save(
        {"ae_config": cfg, "ae_state_dict": ae.state_dict(), "encoder_kind": "local_field_2d_aniso",
         "val_recon_final": 0.01},
        path,
    )
    return cfg, ae


def _save_local_mlp_2d_aniso_propagator_checkpoint(
    path, n_sites_x=4, n_sites_y=8, local_channels=2, token_d_model=6, attn_window=1, forcing_n=4,
):
    d_latent = n_sites_x * n_sites_y * local_channels
    cfg = AuxPropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="local_mlp_2d_aniso", attn_window=attn_window,
        local_mlp2d_aniso_n_sites_x=n_sites_x, local_mlp2d_aniso_n_sites_y=n_sites_y,
        local_mlp2d_aniso_local_channels=local_channels, local_mlp2d_aniso_token_d_model=token_d_model,
        local_mlp2d_aniso_n_layers=2, local_mlp2d_aniso_forcing_n=forcing_n,
    )
    prop = AuxPropagator(cfg)
    torch.save({"prop_config": cfg, "prop_state_dict": prop.state_dict()}, path)
    return cfg, prop


def test_resized_autoencoder_2d_aniso_x_actually_transfers_trained_weights(tmp_path):
    path = tmp_path / "ae.pt"
    old_cfg, ae_orig = _save_local_field_2d_aniso_checkpoint(
        path, n_sites_x=4, n_sites_y=8, NX=32, NY=64, local_channels=2,
    )
    # patch_x = 32/4 = 8 -- double n_sites_x AND NX to keep patch_x fixed; NY/n_sites_y untouched
    ae_new, new_cfg, _ = load_autoencoder_checkpoint_resized_2d_aniso_x(
        str(path), new_n_sites_x=8, new_NX=64,
    )
    assert new_cfg.n_sites_x == 8
    assert new_cfg.NX == 64
    assert (new_cfg.n_sites_y, new_cfg.NY) == (old_cfg.n_sites_y, old_cfg.NY)
    assert new_cfg.local_channels == old_cfg.local_channels
    assert torch.allclose(
        dict(ae_new.named_parameters())["enc_mix.0.weight"],
        dict(ae_orig.named_parameters())["enc_mix.0.weight"],
    )


def test_resized_autoencoder_2d_aniso_x_forward_pass_works_at_new_size(tmp_path):
    path = tmp_path / "ae.pt"
    _save_local_field_2d_aniso_checkpoint(path, n_sites_x=4, n_sites_y=8, NX=32, NY=64, local_channels=2)
    ae_new, new_cfg, _ = load_autoencoder_checkpoint_resized_2d_aniso_x(str(path), new_n_sites_x=8, new_NX=64)
    u = torch.randn(3, 64 * 64)
    z = ae_new.encode(u)
    assert z.shape == (3, 8 * 8 * 2)
    u_hat = ae_new.decode(z)
    assert u_hat.shape == (3, 64 * 64)


def test_resized_autoencoder_2d_aniso_x_phase_buffers_unchanged(tmp_path):
    """The y-phase buffers must be identical before/after resize (not
    just same shape) since they're recomputed fresh from the same
    NY/n_sites_y/forcing_n rather than loaded from the checkpoint."""
    path = tmp_path / "ae.pt"
    _, ae_orig = _save_local_field_2d_aniso_checkpoint(path, n_sites_x=4, n_sites_y=8, NX=32, NY=64, local_channels=2)
    ae_new, _, _ = load_autoencoder_checkpoint_resized_2d_aniso_x(str(path), new_n_sites_x=8, new_NX=64)
    assert torch.allclose(ae_new.cos_y_site, ae_orig.cos_y_site)
    assert torch.allclose(ae_new.sin_y_site, ae_orig.sin_y_site)
    assert torch.allclose(ae_new.cos_y_pixel, ae_orig.cos_y_pixel)
    assert torch.allclose(ae_new.sin_y_pixel, ae_orig.sin_y_pixel)


def test_resized_autoencoder_2d_aniso_x_rejects_mismatched_patch_size(tmp_path):
    path = tmp_path / "ae.pt"
    _save_local_field_2d_aniso_checkpoint(path, n_sites_x=4, n_sites_y=8, NX=32, NY=64, local_channels=2)  # patch_x=8
    with pytest.raises(ValueError, match="patch_x"):
        # n_sites_x doubled but NX only x1.5 -> patch_x changes (32*1.5/8=6 != 8)
        load_autoencoder_checkpoint_resized_2d_aniso_x(str(path), new_n_sites_x=8, new_NX=48)


def test_resized_autoencoder_2d_aniso_x_rejects_non_aniso_encoder_kind(tmp_path):
    path = tmp_path / "ae.pt"
    torch.save({"ae_config": object(), "ae_state_dict": {}, "encoder_kind": "local_field_2d"}, path)
    with pytest.raises(ValueError, match="local_field_2d_aniso"):
        load_autoencoder_checkpoint_resized_2d_aniso_x(str(path), new_n_sites_x=8, new_NX=64)


def test_resized_propagator_2d_aniso_x_actually_transfers_trained_weights(tmp_path):
    path = tmp_path / "prop.pt"
    old_cfg, prop_orig = _save_local_mlp_2d_aniso_propagator_checkpoint(
        path, n_sites_x=4, n_sites_y=8, local_channels=2, token_d_model=6,
    )
    prop_new, new_cfg, _ = load_propagator_checkpoint_resized_2d_aniso_x(str(path), new_n_sites_x=8)
    assert new_cfg.d_latent == 8 * 8 * 2
    assert new_cfg.local_mlp2d_aniso_n_sites_y == old_cfg.local_mlp2d_aniso_n_sites_y
    assert torch.allclose(
        dict(prop_new.named_parameters())["body.blocks.0.mix.weight"],
        dict(prop_orig.named_parameters())["body.blocks.0.mix.weight"],
    )


def test_resized_propagator_2d_aniso_x_forward_pass_works_at_new_size(tmp_path):
    path = tmp_path / "prop.pt"
    _save_local_mlp_2d_aniso_propagator_checkpoint(path, n_sites_x=4, n_sites_y=8, local_channels=2, token_d_model=6)
    prop_new, new_cfg, _ = load_propagator_checkpoint_resized_2d_aniso_x(str(path), new_n_sites_x=8)
    z = torch.randn(3, new_cfg.d_latent)
    out = prop_new.step(z, z)
    assert out.shape == (3, new_cfg.d_latent)


def test_resized_propagator_2d_aniso_x_rejects_non_local_mlp_2d_aniso_backbone(tmp_path):
    path = tmp_path / "prop.pt"
    cfg = AuxPropagatorConfig(d_latent=8, mode="markovian", backbone="mlp")
    prop = AuxPropagator(cfg)
    torch.save({"prop_config": cfg, "prop_state_dict": prop.state_dict()}, path)
    with pytest.raises(ValueError, match="local_mlp_2d_aniso"):
        load_propagator_checkpoint_resized_2d_aniso_x(str(path), new_n_sites_x=8)
