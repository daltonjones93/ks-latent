"""Tests for `encoder_kind="spectral_field"`
(`KSAutoencoderSpectralField`/`SpectralFieldAutoencoderConfig`, added
2026-09-06, see docs/sine_transform_pde_plan.md).
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import MLPAutoencoderConfig, SpectralFieldAutoencoderConfig, ViTAutoencoderConfig
from ks_latent.models import build_autoencoder, load_autoencoder_checkpoint
from ks_latent.models.autoencoder_mlp import KSAutoencoderMLP
from ks_latent.models.autoencoder_spectral_field import KSAutoencoderSpectralField


def _vit_none_cfg(**overrides):
    defaults = dict(
        NX=64, patch_size=2, d_model=16, n_heads=2, n_blocks=1, pool="none", d_latent=32,
        pos_encoding="linear",
    )
    defaults.update(overrides)
    return ViTAutoencoderConfig(**defaults)


def _mlp_inner_cfg(**overrides):
    defaults = dict(NX=64, hidden=(32, 16), d_latent=32)
    defaults.update(overrides)
    return MLPAutoencoderConfig(**defaults)


def test_encode_decode_shapes():
    cfg = SpectralFieldAutoencoderConfig(vit=_vit_none_cfg(), K=8, L=100.0)
    ae = KSAutoencoderSpectralField(cfg)
    u = torch.randn(4, 64)
    z = ae.encode(u)
    assert z.shape == (4, 16)  # 2*K
    u_hat = ae.decode(z)
    assert u_hat.shape == (4, 64)


def test_forward_matches_encode_decode():
    cfg = SpectralFieldAutoencoderConfig(vit=_vit_none_cfg(), K=8, L=100.0)
    ae = KSAutoencoderSpectralField(cfg)
    u = torch.randn(3, 64)
    u_hat, z = ae.forward(u)
    assert torch.allclose(u_hat, ae.decode(ae.encode(u)))
    assert z.shape == (3, 16)


def test_gradient_flows_end_to_end():
    cfg = SpectralFieldAutoencoderConfig(vit=_vit_none_cfg(), K=8, L=100.0)
    ae = KSAutoencoderSpectralField(cfg)
    u = torch.randn(3, 64, requires_grad=True)
    z = ae.encode(u)
    ae.decode(z).sum().backward()
    assert torch.isfinite(u.grad).all()
    assert ae.inner.enc_proj.weight.grad is not None
    assert ae.inner.dec_out.weight.grad is not None


def test_d_latent_property_is_2K_not_vit_d_latent():
    cfg = SpectralFieldAutoencoderConfig(vit=_vit_none_cfg(), K=8, L=100.0)
    assert cfg.d_latent == 16
    assert cfg.vit.d_latent == 32  # N_w, unchanged
    assert cfg.N_w == 32
    assert cfg.NX == 64


def test_pool_local_also_allowed():
    vit_cfg = _vit_none_cfg(pool="local", pool_window=2, d_latent=16)
    cfg = SpectralFieldAutoencoderConfig(vit=vit_cfg, K=4, L=100.0)
    ae = KSAutoencoderSpectralField(cfg)
    u = torch.randn(2, 64)
    z = ae.encode(u)
    assert z.shape == (2, 8)


def test_rejects_pool_mean():
    with pytest.raises(ValueError, match="pool"):
        SpectralFieldAutoencoderConfig(vit=_vit_none_cfg(pool="mean"), K=8, L=100.0)


def test_rejects_pool_token_mlp():
    with pytest.raises(ValueError, match="pool"):
        SpectralFieldAutoencoderConfig(
            vit=_vit_none_cfg(pool="token_mlp", d_model=16), K=8, L=100.0,
        )


def test_rejects_dec_pool_mismatch():
    with pytest.raises(ValueError, match="dec_pool"):
        SpectralFieldAutoencoderConfig(vit=_vit_none_cfg(dec_pool="mean"), K=8, L=100.0)


def test_rejects_local_channels_not_one():
    # pool="local" with pool_window=2 and d_latent=32 -> n_sites=16,
    # local_channels = 32//16 = 2, not a scalar field.
    with pytest.raises(ValueError, match="local_channels"):
        SpectralFieldAutoencoderConfig(
            vit=_vit_none_cfg(pool="local", pool_window=2, d_latent=32), K=8, L=100.0,
        )


def test_rejects_K_out_of_range():
    with pytest.raises(ValueError, match="K="):
        SpectralFieldAutoencoderConfig(vit=_vit_none_cfg(), K=100, L=100.0)
    with pytest.raises(ValueError, match="K="):
        SpectralFieldAutoencoderConfig(vit=_vit_none_cfg(), K=0, L=100.0)


def test_rejects_nonpositive_L():
    with pytest.raises(ValueError, match="L"):
        SpectralFieldAutoencoderConfig(vit=_vit_none_cfg(), K=8, L=0.0)


def test_build_autoencoder_dispatch():
    cfg = SpectralFieldAutoencoderConfig(vit=_vit_none_cfg(), K=8, L=100.0)
    ae = build_autoencoder("spectral_field", cfg)
    assert isinstance(ae, KSAutoencoderSpectralField)


def test_checkpoint_round_trip(tmp_path):
    cfg = SpectralFieldAutoencoderConfig(vit=_vit_none_cfg(), K=8, L=100.0)
    ae = KSAutoencoderSpectralField(cfg)
    path = tmp_path / "ae.pt"
    torch.save({"ae_config": cfg, "ae_state_dict": ae.state_dict(), "encoder_kind": "spectral_field"}, path)
    loaded, loaded_cfg, ckpt = load_autoencoder_checkpoint(str(path))
    assert isinstance(loaded, KSAutoencoderSpectralField)
    u = torch.randn(2, 64)
    assert torch.allclose(loaded.encode(u), ae.encode(u))


# ---- mlp inner model (added 2026-09-09, user-directed: "let the encoder
# be a general mlp ... the vit might not be the right model for this") ----


def test_mlp_inner_requires_exactly_one_of_vit_mlp():
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        SpectralFieldAutoencoderConfig(vit=_vit_none_cfg(), mlp=_mlp_inner_cfg(), K=8, L=100.0)
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        SpectralFieldAutoencoderConfig(K=8, L=100.0)


def test_mlp_inner_encode_decode_shapes():
    cfg = SpectralFieldAutoencoderConfig(mlp=_mlp_inner_cfg(), K=8, L=100.0)
    ae = KSAutoencoderSpectralField(cfg)
    assert isinstance(ae.inner, KSAutoencoderMLP)
    u = torch.randn(4, 64)
    z = ae.encode(u)
    assert z.shape == (4, 16)  # 2*K
    u_hat = ae.decode(z)
    assert u_hat.shape == (4, 64)


def test_mlp_inner_gradient_flows_end_to_end():
    cfg = SpectralFieldAutoencoderConfig(mlp=_mlp_inner_cfg(), K=8, L=100.0)
    ae = KSAutoencoderSpectralField(cfg)
    u = torch.randn(3, 64, requires_grad=True)
    z = ae.encode(u)
    ae.decode(z).sum().backward()
    assert torch.isfinite(u.grad).all()
    assert ae.inner.enc_out.weight.grad is not None
    assert ae.inner.dec_out.weight.grad is not None


def test_mlp_inner_N_w_and_NX_properties():
    cfg = SpectralFieldAutoencoderConfig(mlp=_mlp_inner_cfg(), K=8, L=100.0)
    assert cfg.N_w == 32
    assert cfg.NX == 64
    assert cfg.d_latent == 16


def test_mlp_inner_build_autoencoder_dispatch():
    cfg = SpectralFieldAutoencoderConfig(mlp=_mlp_inner_cfg(), K=8, L=100.0)
    ae = build_autoencoder("spectral_field", cfg)
    assert isinstance(ae, KSAutoencoderSpectralField)
    assert isinstance(ae.inner, KSAutoencoderMLP)
