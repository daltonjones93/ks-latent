"""Tests for `KSAutoencoderViTFourierHybrid`/`ConservedFourierMLP`/
`ViTFourierHybridAutoencoderConfig` (added 2026-09-04, user-directed:
"I don't think the spectral norm is worth pursuing based on what we're
seeing. I want to try the following hybrid network for the encoder and
decoder I want a Vit for function space, the same structure as section 52
plus I want a Fourier mlp that only acts on frequency space ... let's
make the absolute value of each output of each layer sum to the same
value which is the original sum of the absolute values of the frequency
coefficients ... Then just add the output of the ViT and the Fourier
mlp").
"""

from __future__ import annotations

import torch

from ks_latent.config import ViTAutoencoderConfig, ViTFourierHybridAutoencoderConfig
from ks_latent.models import build_autoencoder, load_autoencoder_checkpoint
from ks_latent.models.autoencoder_vit_fourier_hybrid import (
    ConservedFourierMLP,
    KSAutoencoderViTFourierHybrid,
    _fourier_features,
)


def _small_cfg(**overrides) -> ViTFourierHybridAutoencoderConfig:
    vit_kwargs = dict(NX=32, patch_size=4, d_model=8, n_heads=2, n_blocks=1, d_latent=6)
    vit_kwargs.update(overrides.pop("vit_kwargs", {}))
    vit = ViTAutoencoderConfig(**vit_kwargs)
    kwargs = dict(vit=vit, fourier_hidden=16, fourier_blocks=1)
    kwargs.update(overrides)
    return ViTFourierHybridAutoencoderConfig(**kwargs)


def test_encode_decode_shapes():
    cfg = _small_cfg()
    ae = KSAutoencoderViTFourierHybrid(cfg)
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_forward_matches_encode_then_decode():
    torch.manual_seed(0)
    cfg = _small_cfg()
    ae = KSAutoencoderViTFourierHybrid(cfg)
    ae.eval()
    u = torch.randn(3, 32)
    u_hat, z = ae.forward(u)
    assert torch.allclose(u_hat, ae.decode(ae.encode(u)))
    assert torch.allclose(z, ae.encode(u))


def test_gradient_flows_end_to_end():
    cfg = _small_cfg()
    ae = KSAutoencoderViTFourierHybrid(cfg)
    u = torch.randn(4, 32, requires_grad=True)
    u_hat, z = ae.forward(u)
    (u_hat.sum() + z.sum()).backward()
    assert torch.isfinite(u.grad).all()
    for name, p in ae.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name


def test_build_autoencoder_dispatches_to_hybrid():
    cfg = _small_cfg()
    ae = build_autoencoder("vit_fourier_hybrid", cfg)
    assert isinstance(ae, KSAutoencoderViTFourierHybrid)


def test_checkpoint_round_trip(tmp_path):
    cfg = _small_cfg()
    ae = KSAutoencoderViTFourierHybrid(cfg)
    ae.eval()
    u = torch.randn(4, 32)
    z_before = ae.encode(u)

    path = tmp_path / "ae.pt"
    torch.save(
        {"ae_state_dict": ae.state_dict(), "ae_config": cfg, "encoder_kind": "vit_fourier_hybrid"},
        path,
    )
    loaded, loaded_cfg, ckpt = load_autoencoder_checkpoint(str(path))
    assert isinstance(loaded, KSAutoencoderViTFourierHybrid)
    loaded.eval()
    z_after = loaded.encode(u)
    assert torch.allclose(z_before, z_after)


def test_works_under_bfloat16_autocast():
    cfg = _small_cfg()
    ae = KSAutoencoderViTFourierHybrid(cfg)
    u = torch.randn(3, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        z = ae.encode(u)
        u_hat = ae.decode(z)
    assert torch.isfinite(z).all()
    assert torch.isfinite(u_hat).all()


def test_config_d_latent_and_NX_properties_mirror_vit_subconfig():
    cfg = _small_cfg()
    assert cfg.d_latent == cfg.vit.d_latent == 6
    assert cfg.NX == cfg.vit.NX == 32


# ---- ConservedFourierMLP: L1 "conservation law" at every layer ----


def test_conserved_fourier_mlp_output_shape():
    mlp = ConservedFourierMLP(in_modes=5, out_dim=6, hidden=16, n_blocks=2)
    feats = torch.randn(4, 10)  # 2*5
    out = mlp(feats)
    assert out.shape == (4, 6)


def test_conserved_fourier_mlp_output_l1_matches_input_l1():
    """The defining property: the FINAL output's L1 norm exactly equals
    the ORIGINAL input features' L1 norm, per sample."""
    torch.manual_seed(0)
    mlp = ConservedFourierMLP(in_modes=5, out_dim=6, hidden=16, n_blocks=2)
    feats = torch.randn(8, 10) * 3.7  # arbitrary scale
    target_l1 = feats.abs().sum(dim=-1)
    out = mlp(feats)
    out_l1 = out.abs().sum(dim=-1)
    assert torch.allclose(out_l1, target_l1, rtol=1e-4, atol=1e-4)


def test_conserved_fourier_mlp_intermediate_layers_also_conserve():
    """Not just the final output -- every intermediate layer's output
    also satisfies the same L1 constraint (verified by re-implementing
    the forward pass manually and checking each stage)."""
    torch.manual_seed(0)
    mlp = ConservedFourierMLP(in_modes=4, out_dim=5, hidden=12, n_blocks=2)
    feats = torch.randn(6, 8)
    target_l1 = feats.abs().sum(dim=-1, keepdim=True)

    h = mlp._conserve(mlp.act(mlp.input_proj(feats)), target_l1)
    assert torch.allclose(h.abs().sum(dim=-1, keepdim=True), target_l1, rtol=1e-4, atol=1e-4)
    for block in mlp.blocks:
        h = mlp._conserve(mlp.act(block(h)), target_l1)
        assert torch.allclose(h.abs().sum(dim=-1, keepdim=True), target_l1, rtol=1e-4, atol=1e-4)


def test_conserved_fourier_mlp_gradient_flows():
    mlp = ConservedFourierMLP(in_modes=5, out_dim=6, hidden=16, n_blocks=1)
    feats = torch.randn(4, 10, requires_grad=True)
    out = mlp(feats)
    out.sum().backward()
    assert torch.isfinite(feats.grad).all()


def test_fourier_features_shape_and_finite():
    x = torch.randn(5, 32)
    feats = _fourier_features(x, n_modes=7)
    assert feats.shape == (5, 14)
    assert torch.isfinite(feats).all()


# ---- fourier_kind="ifft": Section 75's own FourierIFFTBody mechanism
# instead of ConservedFourierMLP -- added 2026-09-04, user-directed
# (Section 81): "can we have the same kind of hybrid vit fourier mlp,
# just using the exact fourier mlp from 75 except only using the
# frequency components". No raw-value/masked path at all, unlike Section
# 75's own AE (which sums a masked raw path alongside FourierIFFTBody). ----


def test_ifft_kind_uses_fourier_ifft_body():
    from ks_latent.models.propagator import FourierIFFTBody

    cfg = _small_cfg(fourier_kind="ifft")
    ae = KSAutoencoderViTFourierHybrid(cfg)
    assert isinstance(ae.fourier_encoder, FourierIFFTBody)
    assert isinstance(ae.fourier_decoder, FourierIFFTBody)


def test_ifft_kind_shapes_and_gradient_flow():
    cfg = _small_cfg(fourier_kind="ifft")
    ae = KSAutoencoderViTFourierHybrid(cfg)
    u = torch.randn(5, 32, requires_grad=True)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)
    (u_hat.sum() + z.sum()).backward()
    assert torch.isfinite(u.grad).all()


def test_ifft_kind_works_under_bfloat16_autocast():
    cfg = _small_cfg(fourier_kind="ifft")
    ae = KSAutoencoderViTFourierHybrid(cfg)
    u = torch.randn(3, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        z = ae.encode(u)
        u_hat = ae.decode(z)
    assert torch.isfinite(z).all()
    assert torch.isfinite(u_hat).all()


def test_ifft_kind_checkpoint_round_trip(tmp_path):
    cfg = _small_cfg(fourier_kind="ifft")
    ae = KSAutoencoderViTFourierHybrid(cfg)
    ae.eval()
    u = torch.randn(4, 32)
    z_before = ae.encode(u)

    path = tmp_path / "ae.pt"
    torch.save(
        {"ae_state_dict": ae.state_dict(), "ae_config": cfg, "encoder_kind": "vit_fourier_hybrid"},
        path,
    )
    loaded, loaded_cfg, ckpt = load_autoencoder_checkpoint(str(path))
    assert isinstance(loaded, KSAutoencoderViTFourierHybrid)
    loaded.eval()
    z_after = loaded.encode(u)
    assert torch.allclose(z_before, z_after)


def test_conserved_is_still_default():
    cfg = _small_cfg()
    assert cfg.fourier_kind == "conserved"
    ae = KSAutoencoderViTFourierHybrid(cfg)
    assert isinstance(ae.fourier_encoder, ConservedFourierMLP)


def test_invalid_fourier_kind_rejected():
    import pytest

    with pytest.raises(ValueError):
        _small_cfg(fourier_kind="bogus")


# ---- enc_out_modes/dec_out_modes (added 2026-09-05, user-directed:
# "can we explicitly penalize higher frequency terms in the irfft matrix?
# or even truncate these completely?") ----


def test_enc_out_modes_truncates_fourier_encoder_only():
    from ks_latent.models.propagator import FourierIFFTBody

    cfg = _small_cfg(fourier_kind="ifft", enc_out_modes=2)
    ae = KSAutoencoderViTFourierHybrid(cfg)
    assert isinstance(ae.fourier_encoder, FourierIFFTBody)
    assert ae.fourier_encoder.out_modes == 2
    # decoder untouched (default None -> full spectrum; its out_len is NX=32)
    assert ae.fourier_decoder.out_modes == 32 // 2 + 1


def test_enc_out_modes_shapes_and_gradient_flow():
    cfg = _small_cfg(fourier_kind="ifft", enc_out_modes=2)
    ae = KSAutoencoderViTFourierHybrid(cfg)
    u = torch.randn(5, 32, requires_grad=True)
    z = ae.encode(u)
    assert z.shape == (5, 6)  # irfft still outputs full d_latent length regardless of out_modes
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)
    (u_hat.sum() + z.sum()).backward()
    assert torch.isfinite(u.grad).all()


def test_dec_out_modes_truncates_fourier_decoder_only():
    cfg = _small_cfg(fourier_kind="ifft", dec_out_modes=3)
    ae = KSAutoencoderViTFourierHybrid(cfg)
    assert ae.fourier_decoder.out_modes == 3
    assert ae.fourier_encoder.out_modes == 6 // 2 + 1  # encoder untouched (its out_len is d_latent=6)


def test_out_modes_none_by_default_matches_full_spectrum():
    cfg = _small_cfg(fourier_kind="ifft")
    ae = KSAutoencoderViTFourierHybrid(cfg)
    assert ae.fourier_encoder.out_modes == 6 // 2 + 1
    assert ae.fourier_decoder.out_modes == 32 // 2 + 1


def test_out_modes_rejected_under_conserved_kind():
    import pytest

    with pytest.raises(ValueError, match="fourier_kind='ifft'"):
        _small_cfg(enc_out_modes=2)  # default fourier_kind="conserved"
