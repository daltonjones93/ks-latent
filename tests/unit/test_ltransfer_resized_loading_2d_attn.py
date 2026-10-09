"""Tests for `load_autoencoder_checkpoint_resized_2d_attn`
(`ks_latent/models/__init__.py`, added 2026-09-29, user-directed: "please
implement the local attention variant you described above"). Mirrors
`test_ltransfer_resized_loading_2d.py`'s discipline: tiny synthetic
checkpoint in a temp dir, verify resized loading actually transfers
trained weight VALUES (not just shapes), rejects mismatched cases, and
-- specific to this architecture -- that the structural mask/index
buffers are recomputed fresh at the NEW size rather than loaded stale.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import LocalAttentionAutoencoderConfig2D
from ks_latent.models import load_autoencoder_checkpoint_resized_2d_attn
from ks_latent.models.autoencoder_local_attention_2d import KSAutoencoderLocalAttention2D


def _save_checkpoint(
    path, n_sites_x=4, n_sites_y=4, NX=32, NY=32, local_channels=2,
    window=1, n_site_mix_layers=2, d_model=8, n_heads=2, mlp_ratio=2,
):
    cfg = LocalAttentionAutoencoderConfig2D(
        NX=NX, NY=NY, n_sites_x=n_sites_x, n_sites_y=n_sites_y, local_channels=local_channels,
        window=window, n_site_mix_layers=n_site_mix_layers, d_model=d_model, n_heads=n_heads,
        mlp_ratio=mlp_ratio,
    )
    ae = KSAutoencoderLocalAttention2D(cfg)
    torch.save(
        {"ae_config": cfg, "ae_state_dict": ae.state_dict(), "encoder_kind": "local_attention_2d",
         "val_recon_final": 0.01},
        path,
    )
    return cfg, ae


def test_resized_autoencoder_actually_transfers_trained_weights(tmp_path):
    path = tmp_path / "ae.pt"
    old_cfg, ae_orig = _save_checkpoint(path, n_sites_x=4, n_sites_y=4, NX=32, NY=32, local_channels=2)
    # patch_x=patch_y=8 -- double sites AND NX/NY to keep patch size fixed
    ae_new, new_cfg, _ = load_autoencoder_checkpoint_resized_2d_attn(
        str(path), new_n_sites_x=8, new_n_sites_y=8, new_NX=64, new_NY=64,
    )
    assert (new_cfg.n_sites_x, new_cfg.n_sites_y) == (8, 8)
    assert (new_cfg.NX, new_cfg.NY) == (64, 64)
    assert new_cfg.local_channels == old_cfg.local_channels
    assert torch.allclose(
        dict(ae_new.named_parameters())["enc_blocks.0.qkv.weight"],
        dict(ae_orig.named_parameters())["enc_blocks.0.qkv.weight"],
    )
    assert torch.allclose(
        dict(ae_new.named_parameters())["enc_blocks.0.bias_table"],
        dict(ae_orig.named_parameters())["enc_blocks.0.bias_table"],
    )


def test_resized_autoencoder_forward_pass_works_at_new_size(tmp_path):
    path = tmp_path / "ae.pt"
    _save_checkpoint(path, n_sites_x=4, n_sites_y=4, NX=32, NY=32, local_channels=2)
    ae_new, new_cfg, _ = load_autoencoder_checkpoint_resized_2d_attn(
        str(path), new_n_sites_x=8, new_n_sites_y=8, new_NX=64, new_NY=64,
    )
    u = torch.randn(3, 64 * 64)
    z = ae_new.encode(u)
    assert z.shape == (3, 8 * 8 * 2)
    u_hat = ae_new.decode(z)
    assert u_hat.shape == (3, 64 * 64)


def test_resized_autoencoder_mask_buffers_match_new_size_not_old(tmp_path):
    """The structural mask/index buffers must be recomputed for the NEW
    grid -- checking this directly is the point of persistent=False."""
    path = tmp_path / "ae.pt"
    _save_checkpoint(path, n_sites_x=4, n_sites_y=4, NX=32, NY=32, local_channels=2)
    ae_new, new_cfg, _ = load_autoencoder_checkpoint_resized_2d_attn(
        str(path), new_n_sites_x=8, new_n_sites_y=8, new_NX=64, new_NY=64,
    )
    N_new = 8 * 8
    assert ae_new.enc_blocks[0].mask.shape == (N_new, N_new)


def test_resized_autoencoder_rejects_mismatched_patch_size(tmp_path):
    path = tmp_path / "ae.pt"
    _save_checkpoint(path, n_sites_x=4, n_sites_y=4, NX=32, NY=32, local_channels=2)  # patch=8
    with pytest.raises(ValueError, match="patch size"):
        # n_sites doubled but NX only x1.5 -> patch size changes (32*1.5/8=6 != 8)
        load_autoencoder_checkpoint_resized_2d_attn(str(path), new_n_sites_x=8, new_n_sites_y=8, new_NX=48, new_NY=48)


def test_resized_autoencoder_rejects_non_local_attention_2d_encoder(tmp_path):
    path = tmp_path / "ae.pt"
    torch.save({"ae_config": object(), "ae_state_dict": {}, "encoder_kind": "local_field_2d"}, path)
    with pytest.raises(ValueError, match="local_attention_2d"):
        load_autoencoder_checkpoint_resized_2d_attn(str(path), new_n_sites_x=8, new_n_sites_y=8, new_NX=64, new_NY=64)
