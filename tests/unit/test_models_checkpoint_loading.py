"""Tests for `ks_latent.models.load_autoencoder_checkpoint`/`build_autoencoder`,
added 2026-08-29 after catching a real bug: `run_analysis_suite.py`,
`run_da_pff.py`, and `run_diagnostics.py` all hardcoded `KSAutoencoderPatched`
when loading a Stage-1 checkpoint, regardless of which architecture it was
actually trained with -- silently wrong for any `--encoder mlp`/`--encoder
vit` checkpoint.
"""

from __future__ import annotations

import torch

from ks_latent.config import AutoencoderConfig, MLPAutoencoderConfig, ViTAutoencoderConfig
from ks_latent.models import build_autoencoder, load_autoencoder_checkpoint
from ks_latent.models.autoencoder_mlp import KSAutoencoderMLP
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.autoencoder_vit import KSAutoencoderViT


def test_build_autoencoder_dispatches_on_encoder_kind():
    assert isinstance(build_autoencoder("mlp", MLPAutoencoderConfig(NX=16, d_latent=4)), KSAutoencoderMLP)
    assert isinstance(
        build_autoencoder("vit", ViTAutoencoderConfig(NX=16, patch_size=4, d_latent=4)), KSAutoencoderViT
    )
    assert isinstance(
        build_autoencoder(
            "transformer",
            AutoencoderConfig(NX=16, patch_size=4, group_size=2, d_model=8, nhead=2, dim_ff=8,
                               patch_embed_hidden=8, n_local_layers=1, n_global_layers=1,
                               n_query_tokens=2, d_latent=4),
        ),
        KSAutoencoderPatched,
    )


def test_build_autoencoder_defaults_unknown_kind_to_transformer():
    cfg = AutoencoderConfig(NX=16, patch_size=4, group_size=2, d_model=8, nhead=2, dim_ff=8,
                             patch_embed_hidden=8, n_local_layers=1, n_global_layers=1,
                             n_query_tokens=2, d_latent=4)
    assert isinstance(build_autoencoder("something_new", cfg), KSAutoencoderPatched)
    assert isinstance(build_autoencoder("", cfg), KSAutoencoderPatched)


def test_load_autoencoder_checkpoint_roundtrips_mlp(tmp_path):
    cfg = MLPAutoencoderConfig(NX=16, hidden=(32, 16), d_latent=4)
    ae = KSAutoencoderMLP(cfg)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"ae_state_dict": ae.state_dict(), "ae_config": cfg, "encoder_kind": "mlp"}, ckpt_path)

    loaded, loaded_cfg, ckpt = load_autoencoder_checkpoint(ckpt_path)
    assert isinstance(loaded, KSAutoencoderMLP)
    assert loaded_cfg == cfg
    u = torch.randn(3, 16)
    assert torch.allclose(loaded.encode(u), ae.encode(u))


def test_load_autoencoder_checkpoint_roundtrips_vit(tmp_path):
    cfg = ViTAutoencoderConfig(NX=16, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=4)
    ae = KSAutoencoderViT(cfg)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"ae_state_dict": ae.state_dict(), "ae_config": cfg, "encoder_kind": "vit"}, ckpt_path)

    loaded, loaded_cfg, ckpt = load_autoencoder_checkpoint(ckpt_path)
    assert isinstance(loaded, KSAutoencoderViT)
    u = torch.randn(3, 16)
    assert torch.allclose(loaded.encode(u), ae.encode(u))


def test_load_autoencoder_checkpoint_missing_encoder_kind_defaults_to_transformer(tmp_path):
    """Checkpoints written before `encoder_kind` existed (pre-2026-08-29)
    must still load correctly as the patched-transformer."""
    cfg = AutoencoderConfig(NX=16, patch_size=4, group_size=2, d_model=8, nhead=2, dim_ff=8,
                             patch_embed_hidden=8, n_local_layers=1, n_global_layers=1,
                             n_query_tokens=2, d_latent=4)
    ae = KSAutoencoderPatched(cfg)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"ae_state_dict": ae.state_dict(), "ae_config": cfg}, ckpt_path)  # no encoder_kind key

    loaded, _, _ = load_autoencoder_checkpoint(ckpt_path)
    assert isinstance(loaded, KSAutoencoderPatched)
