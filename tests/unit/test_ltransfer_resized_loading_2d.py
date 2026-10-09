"""Tests for `load_autoencoder_checkpoint_resized_2d`/`load_propagator_
checkpoint_resized_2d` (`ks_latent/models/__init__.py`, added 2026-09-29,
G1/G2: `docs/ks_local_field_ltransfer.tex` "L-transfer for the 2D
systems" plan) -- 2D analogue of `test_ltransfer_resized_loading.py`,
same discipline: tiny synthetic checkpoints in a temp dir, verify resized
loading actually transfers trained weight VALUES (not just shapes) and
rejects the cases it should.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import AuxPropagatorConfig, LocalFieldAutoencoderConfig2D
from ks_latent.models import (
    load_autoencoder_checkpoint_resized_2d,
    load_propagator_checkpoint_resized_2d,
)
from ks_latent.models.autoencoder_local_field_2d import KSAutoencoderLocalField2D
from ks_latent.models.propagator import AuxPropagator


def _save_local_field_2d_checkpoint(
    path, n_sites_x=4, n_sites_y=4, NX=32, NY=32, local_channels=2,
    site_mix_radius=1, n_site_mix_layers=2, hidden=6,
):
    cfg = LocalFieldAutoencoderConfig2D(
        NX=NX, NY=NY, n_sites_x=n_sites_x, n_sites_y=n_sites_y, local_channels=local_channels,
        site_mix_radius=site_mix_radius, n_site_mix_layers=n_site_mix_layers, hidden=hidden,
    )
    ae = KSAutoencoderLocalField2D(cfg)
    torch.save(
        {"ae_config": cfg, "ae_state_dict": ae.state_dict(), "encoder_kind": "local_field_2d",
         "val_recon_final": 0.01},
        path,
    )
    return cfg, ae


def _save_local_mlp_2d_propagator_checkpoint(
    path, n_sites_x=4, n_sites_y=4, local_channels=2, token_d_model=6, attn_window=1,
):
    d_latent = n_sites_x * n_sites_y * local_channels
    cfg = AuxPropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="local_mlp_2d", attn_window=attn_window,
        local_mlp2d_n_sites_x=n_sites_x, local_mlp2d_n_sites_y=n_sites_y,
        local_mlp2d_local_channels=local_channels, local_mlp2d_token_d_model=token_d_model,
        local_mlp2d_n_layers=2,
    )
    prop = AuxPropagator(cfg)
    torch.save({"prop_config": cfg, "prop_state_dict": prop.state_dict()}, path)
    return cfg, prop


def test_resized_autoencoder_2d_actually_transfers_trained_weights(tmp_path):
    path = tmp_path / "ae.pt"
    old_cfg, ae_orig = _save_local_field_2d_checkpoint(path, n_sites_x=4, n_sites_y=4, NX=32, NY=32, local_channels=2)
    # patch_x=patch_y=8 -- double sites AND NX/NY to keep patch size fixed
    ae_new, new_cfg, _ = load_autoencoder_checkpoint_resized_2d(
        str(path), new_n_sites_x=8, new_n_sites_y=8, new_NX=64, new_NY=64,
    )
    assert (new_cfg.n_sites_x, new_cfg.n_sites_y) == (8, 8)
    assert (new_cfg.NX, new_cfg.NY) == (64, 64)
    assert new_cfg.local_channels == old_cfg.local_channels
    assert torch.allclose(
        dict(ae_new.named_parameters())["enc_mix.0.weight"],
        dict(ae_orig.named_parameters())["enc_mix.0.weight"],
    )


def test_resized_autoencoder_2d_forward_pass_works_at_new_size(tmp_path):
    path = tmp_path / "ae.pt"
    _save_local_field_2d_checkpoint(path, n_sites_x=4, n_sites_y=4, NX=32, NY=32, local_channels=2)
    ae_new, new_cfg, _ = load_autoencoder_checkpoint_resized_2d(
        str(path), new_n_sites_x=8, new_n_sites_y=8, new_NX=64, new_NY=64,
    )
    u = torch.randn(3, 64 * 64)
    z = ae_new.encode(u)
    assert z.shape == (3, 8 * 8 * 2)
    u_hat = ae_new.decode(z)
    assert u_hat.shape == (3, 64 * 64)


def test_resized_autoencoder_2d_rejects_mismatched_patch_size(tmp_path):
    path = tmp_path / "ae.pt"
    _save_local_field_2d_checkpoint(path, n_sites_x=4, n_sites_y=4, NX=32, NY=32, local_channels=2)  # patch=8
    with pytest.raises(ValueError, match="patch size"):
        # n_sites doubled but NX only x1.5 -> patch size changes (32*1.5/8=6 != 8)
        load_autoencoder_checkpoint_resized_2d(str(path), new_n_sites_x=8, new_n_sites_y=8, new_NX=48, new_NY=48)


def test_resized_autoencoder_2d_rejects_non_local_field_2d_encoder(tmp_path):
    path = tmp_path / "ae.pt"
    torch.save({"ae_config": object(), "ae_state_dict": {}, "encoder_kind": "local_field"}, path)
    with pytest.raises(ValueError, match="local_field_2d"):
        load_autoencoder_checkpoint_resized_2d(str(path), new_n_sites_x=8, new_n_sites_y=8, new_NX=64, new_NY=64)


def test_resized_propagator_2d_actually_transfers_trained_weights(tmp_path):
    path = tmp_path / "prop.pt"
    old_cfg, prop_orig = _save_local_mlp_2d_propagator_checkpoint(
        path, n_sites_x=4, n_sites_y=4, local_channels=2, token_d_model=6,
    )
    prop_new, new_cfg, _ = load_propagator_checkpoint_resized_2d(str(path), new_n_sites_x=8, new_n_sites_y=8)
    assert new_cfg.d_latent == 8 * 8 * 2
    assert torch.allclose(
        dict(prop_new.named_parameters())["body.blocks.0.mix.weight"],
        dict(prop_orig.named_parameters())["body.blocks.0.mix.weight"],
    )


def test_resized_propagator_2d_forward_pass_works_at_new_size(tmp_path):
    path = tmp_path / "prop.pt"
    _save_local_mlp_2d_propagator_checkpoint(path, n_sites_x=4, n_sites_y=4, local_channels=2, token_d_model=6)
    prop_new, new_cfg, _ = load_propagator_checkpoint_resized_2d(str(path), new_n_sites_x=8, new_n_sites_y=8)
    z = torch.randn(3, new_cfg.d_latent)
    out = prop_new.step(z, z)
    assert out.shape == (3, new_cfg.d_latent)


def test_resized_propagator_2d_rejects_non_local_mlp_2d_backbone(tmp_path):
    path = tmp_path / "prop.pt"
    cfg = AuxPropagatorConfig(d_latent=8, mode="markovian", backbone="mlp")
    prop = AuxPropagator(cfg)
    torch.save({"prop_config": cfg, "prop_state_dict": prop.state_dict()}, path)
    with pytest.raises(ValueError, match="local_mlp_2d"):
        load_propagator_checkpoint_resized_2d(str(path), new_n_sites_x=8, new_n_sites_y=8)
