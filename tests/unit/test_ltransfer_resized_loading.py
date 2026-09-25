"""Tests for `load_autoencoder_checkpoint_resized`/`load_propagator_
checkpoint_resized` (`ks_latent/models/__init__.py`, added 2026-09-25,
Phase F2/F3 `docs/steps_4-3.md`, user-directed: "run Phase F and run a
larger L without retraining. Make certain that if L gets larger, the
number of samples gets larger though so the effective sample width
remains the same").

Builds tiny synthetic checkpoints in a temp dir (not depending on real
trained artifacts) matching the exact on-disk format `train_stage1_
patched.py`/`train_stage2_patched.py` write, then verifies resized
loading actually transfers the trained weights (not just constructs a
fresh, differently-initialized model of the right shape) and rejects
the cases it should.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import LocalFieldAutoencoderConfig, PropagatorConfig
from ks_latent.models import (
    load_autoencoder_checkpoint_resized,
    load_propagator_checkpoint_resized,
)
from ks_latent.models.autoencoder_local_field import KSAutoencoderLocalField
from ks_latent.models.propagator import LatentPropagator


def _save_local_field_checkpoint(path, n_sites=4, NX=32, local_channels=2, site_mix_radius=1, n_site_mix_layers=2, hidden=6):
    cfg = LocalFieldAutoencoderConfig(
        NX=NX, n_sites=n_sites, local_channels=local_channels,
        site_mix_radius=site_mix_radius, n_site_mix_layers=n_site_mix_layers, hidden=hidden,
    )
    ae = KSAutoencoderLocalField(cfg)
    torch.save(
        {"ae_config": cfg, "ae_state_dict": ae.state_dict(), "encoder_kind": "local_field",
         "val_recon_final": 0.01},
        path,
    )
    return cfg, ae


def _save_local_mlp_propagator_checkpoint(path, d_latent=8, n_tokens=4, token_d_model=6, attn_window=1):
    cfg = PropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="local_mlp",
        n_tokens=n_tokens, token_d_model=token_d_model, token_n_layers=2, attn_window=attn_window,
        zero_init=False,
    )
    prop = LatentPropagator(cfg)
    torch.save({"prop_config": cfg, "prop_state_dict": prop.state_dict()}, path)
    return cfg, prop


def test_resized_autoencoder_actually_transfers_trained_weights(tmp_path):
    path = tmp_path / "ae.pt"
    old_cfg, ae_orig = _save_local_field_checkpoint(path, n_sites=4, NX=32, local_channels=2)
    # patch_size = NX/n_sites = 8 -- double n_sites AND NX to keep patch_size fixed
    ae_new, new_cfg, ckpt = load_autoencoder_checkpoint_resized(str(path), new_n_sites=8, new_NX=64)
    assert new_cfg.n_sites == 8
    assert new_cfg.NX == 64
    assert new_cfg.local_channels == old_cfg.local_channels
    # weight VALUES, not just shapes, must match the original checkpoint
    assert torch.allclose(
        dict(ae_new.named_parameters())["enc_mix.0.weight"],
        dict(ae_orig.named_parameters())["enc_mix.0.weight"],
    )


def test_resized_autoencoder_forward_pass_works_at_new_size(tmp_path):
    path = tmp_path / "ae.pt"
    _save_local_field_checkpoint(path, n_sites=4, NX=32, local_channels=2)
    ae_new, new_cfg, _ = load_autoencoder_checkpoint_resized(str(path), new_n_sites=8, new_NX=64)
    u = torch.randn(3, 64)
    z = ae_new.encode(u)
    assert z.shape == (3, 8 * 2)
    u_hat = ae_new.decode(z)
    assert u_hat.shape == (3, 64)


def test_resized_autoencoder_rejects_mismatched_patch_size(tmp_path):
    path = tmp_path / "ae.pt"
    _save_local_field_checkpoint(path, n_sites=4, NX=32, local_channels=2)  # patch_size=8
    with pytest.raises(ValueError, match="patch_size"):
        # n_sites doubled but NX only x1.5 -> patch_size changes (32*1.5/8=6 != 8)
        load_autoencoder_checkpoint_resized(str(path), new_n_sites=8, new_NX=48)


def test_resized_autoencoder_rejects_non_local_field_encoder(tmp_path):
    path = tmp_path / "ae.pt"
    torch.save({"ae_config": object(), "ae_state_dict": {}, "encoder_kind": "vit"}, path)
    with pytest.raises(ValueError, match="local_field"):
        load_autoencoder_checkpoint_resized(str(path), new_n_sites=8, new_NX=64)


def test_resized_propagator_actually_transfers_trained_weights(tmp_path):
    path = tmp_path / "prop.pt"
    old_cfg, prop_orig = _save_local_mlp_propagator_checkpoint(path, d_latent=8, n_tokens=4, token_d_model=6)
    # chunk_size = d_latent/n_tokens = 2 -- double both to keep chunk_size fixed
    prop_new, new_cfg, _ = load_propagator_checkpoint_resized(str(path), new_d_latent=16, new_n_tokens=8)
    assert new_cfg.d_latent == 16
    assert new_cfg.n_tokens == 8
    assert torch.allclose(
        dict(prop_new.named_parameters())["body.token_embed.weight"],
        dict(prop_orig.named_parameters())["body.token_embed.weight"],
    )


def test_resized_propagator_forward_pass_works_at_new_size(tmp_path):
    path = tmp_path / "prop.pt"
    _save_local_mlp_propagator_checkpoint(path, d_latent=8, n_tokens=4, token_d_model=6)
    prop_new, new_cfg, _ = load_propagator_checkpoint_resized(str(path), new_d_latent=16, new_n_tokens=8)
    z = torch.randn(3, 16)
    out = prop_new.step_one(z)
    assert out.shape == (3, 16)


def test_resized_propagator_rejects_non_local_mlp_backbone(tmp_path):
    path = tmp_path / "prop.pt"
    cfg = PropagatorConfig(d_latent=8, mode="markovian", backbone="mlp")
    prop = LatentPropagator(cfg)
    torch.save({"prop_config": cfg, "prop_state_dict": prop.state_dict()}, path)
    with pytest.raises(ValueError, match="local_mlp"):
        load_propagator_checkpoint_resized(str(path), new_d_latent=16, new_n_tokens=8)


def test_resized_propagator_rejects_mismatched_chunk_size(tmp_path):
    path = tmp_path / "prop.pt"
    _save_local_mlp_propagator_checkpoint(path, d_latent=8, n_tokens=4, token_d_model=6)  # chunk_size=2
    with pytest.raises(ValueError, match="chunk_size"):
        load_propagator_checkpoint_resized(str(path), new_d_latent=16, new_n_tokens=4)  # chunk_size=4 != 2
