"""Tests for `KSAutoencoderFourierMLP`/`FourierMLPAutoencoderConfig`
(added 2026-09-03, user-directed: "an mlp model that takes in fourier
features ... as the encoder, decoder and propagator"). See
`FourierMLPAutoencoderConfig`'s docstring: raw values concatenated with a
fixed real/imag rfft featurization, fed through `MLPDeltaBody` (the same
class propagator.py's `backbone="mlp"`/`"fourier_mlp"` use).
"""

from __future__ import annotations

import torch

from ks_latent.config import FourierMLPAutoencoderConfig
from ks_latent.models import build_autoencoder, load_autoencoder_checkpoint
from ks_latent.models.autoencoder_fourier_mlp import (
    KSAutoencoderFourierMLP,
    _inverse_fourier_features,
)


def test_encode_decode_shapes():
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=6, hidden=16, n_blocks=1)
    ae = KSAutoencoderFourierMLP(cfg)
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_forward_matches_encode_then_decode():
    torch.manual_seed(0)
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=6, hidden=16, n_blocks=1)
    ae = KSAutoencoderFourierMLP(cfg)
    ae.eval()
    u = torch.randn(3, 32)
    u_hat, z = ae.forward(u)
    assert torch.allclose(u_hat, ae.decode(ae.encode(u)))
    assert torch.allclose(z, ae.encode(u))


def test_default_fno_modes_uses_full_spectrum():
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=6, hidden=16, n_blocks=1)
    ae = KSAutoencoderFourierMLP(cfg)
    assert ae.enc_modes == 32 // 2 + 1
    assert ae.dec_modes == 6 // 2 + 1


def test_fno_modes_clips_to_max_available():
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=6, hidden=16, n_blocks=1, enc_fno_modes=1000, dec_fno_modes=1000,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    assert ae.enc_modes == 32 // 2 + 1
    assert ae.dec_modes == 6 // 2 + 1


def test_build_autoencoder_dispatches_to_fourier_mlp():
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=6, hidden=16, n_blocks=1)
    ae = build_autoencoder("fourier_mlp", cfg)
    assert isinstance(ae, KSAutoencoderFourierMLP)


def test_checkpoint_round_trip(tmp_path):
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=6, hidden=16, n_blocks=1)
    ae = KSAutoencoderFourierMLP(cfg)
    ae.eval()
    u = torch.randn(4, 32)
    z_before = ae.encode(u)

    path = tmp_path / "ae.pt"
    torch.save(
        {"ae_state_dict": ae.state_dict(), "ae_config": cfg, "encoder_kind": "fourier_mlp"},
        path,
    )
    loaded, loaded_cfg, ckpt = load_autoencoder_checkpoint(str(path))
    assert isinstance(loaded, KSAutoencoderFourierMLP)
    loaded.eval()
    z_after = loaded.encode(u)
    assert torch.allclose(z_before, z_after)


def test_works_under_bfloat16_autocast():
    """Regression test (same bug class as SpectralConv1d/logdet_barrier_loss/
    _FourierMLPHistoryDeltaBody): torch.fft.rfft doesn't support bfloat16,
    so the module's own Fourier-feature computation must explicitly cast
    to float32 itself."""
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=6, hidden=16, n_blocks=1)
    ae = KSAutoencoderFourierMLP(cfg)
    u = torch.randn(3, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        z = ae.encode(u)
        u_hat = ae.decode(z)
    assert torch.isfinite(z).all()
    assert torch.isfinite(u_hat).all()


# ---- masked option (attn_window set): masked raw-value path + dense
# Fourier path, added 2026-09-03, user-directed: "use all the fourier
# coefficients, but only let real variables interact with their
# neighbors" -- extended from the propagator to the encoder/decoder. ----


def test_masked_encode_decode_shapes():
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=2)
    ae = KSAutoencoderFourierMLP(cfg)
    assert ae.encoder is None and ae.encoder_masked is not None and ae.encoder_fourier is not None
    assert ae.decoder is None and ae.decoder_masked is not None and ae.decoder_fourier is not None
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 8)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_unmasked_by_default():
    """attn_window=None (default) keeps the original single-dense-body
    structure -- backward compatible."""
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=8, hidden=16, n_blocks=1)
    ae = KSAutoencoderFourierMLP(cfg)
    assert ae.encoder is not None and ae.encoder_masked is None
    assert ae.decoder is not None and ae.decoder_masked is None


def test_masked_encoder_cross_weight_is_exactly_zero_outside_band():
    """Direct test of the masking claim on the rectangular NX<->d_latent
    crossing layer."""
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=2)
    ae = KSAutoencoderFourierMLP(cfg)
    W = ae.encoder_masked.cross.linear.weight.detach()  # (8, 32)
    out_pos = torch.arange(8).float() / 8
    in_pos = torch.arange(32).float() / 32
    diff = (out_pos.unsqueeze(1) - in_pos.unsqueeze(0)).abs()
    diff = torch.minimum(diff, 1.0 - diff)
    in_band = diff <= (2 / 8)
    assert (W[~in_band] == 0).all()
    assert W[in_band].abs().max() > 0


def test_masked_decoder_cross_weight_is_exactly_zero_outside_band():
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=2)
    ae = KSAutoencoderFourierMLP(cfg)
    W = ae.decoder_masked.cross.linear.weight.detach()  # (32, 8)
    out_pos = torch.arange(32).float() / 32
    in_pos = torch.arange(8).float() / 8
    diff = (out_pos.unsqueeze(1) - in_pos.unsqueeze(0)).abs()
    diff = torch.minimum(diff, 1.0 - diff)
    in_band = diff <= (2 / 8)
    assert (W[~in_band] == 0).all()
    assert W[in_band].abs().max() > 0


def test_masked_fourier_paths_stay_fully_dense():
    """"use all the fourier coefficients" -- the Fourier paths must NOT
    be masked, unlike the raw-value paths."""
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=2)
    ae = KSAutoencoderFourierMLP(cfg)
    assert isinstance(ae.encoder_fourier.input_proj, torch.nn.Linear)
    assert not hasattr(ae.encoder_fourier.input_proj, "mask")
    assert isinstance(ae.decoder_fourier.input_proj, torch.nn.Linear)
    assert not hasattr(ae.decoder_fourier.input_proj, "mask")


def test_masked_gradient_flows_end_to_end():
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=2)
    ae = KSAutoencoderFourierMLP(cfg)
    u = torch.randn(4, 32, requires_grad=True)
    u_hat, z = ae.forward(u)
    (u_hat.sum() + z.sum()).backward()
    assert torch.isfinite(u.grad).all()


def test_masked_works_under_bfloat16_autocast():
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=2)
    ae = KSAutoencoderFourierMLP(cfg)
    u = torch.randn(3, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        z = ae.encode(u)
        u_hat = ae.decode(z)
    assert torch.isfinite(z).all()
    assert torch.isfinite(u_hat).all()


def test_masked_checkpoint_round_trip(tmp_path):
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=2)
    ae = KSAutoencoderFourierMLP(cfg)
    ae.eval()
    u = torch.randn(4, 32)
    z_before = ae.encode(u)

    path = tmp_path / "ae.pt"
    torch.save(
        {"ae_state_dict": ae.state_dict(), "ae_config": cfg, "encoder_kind": "fourier_mlp"},
        path,
    )
    loaded, loaded_cfg, ckpt = load_autoencoder_checkpoint(str(path))
    assert isinstance(loaded, KSAutoencoderFourierMLP)
    loaded.eval()
    z_after = loaded.encode(u)
    assert torch.allclose(z_before, z_after)


# ---- dec_use_ifft option: decoder uses irfft-based features instead of
# rfft-based ones, added 2026-09-03, user-directed: "a dense inverse
# fourier mlp for the decoder where the inverse fourier mlp applied the
# ifft not the fft". Always forces the decoder to the fully-dense
# structure regardless of attn_window (which then applies to the encoder
# only) -- so this is also the first coverage of asymmetric
# encoder-masked/decoder-dense construction. ----


def test_inverse_fourier_features_shape_and_finite():
    z = torch.randn(5, 8)
    out = _inverse_fourier_features(z, out_len=32)
    assert out.shape == (5, 32)
    assert torch.isfinite(out).all()


def test_dec_use_ifft_decoder_is_dense_even_with_attn_window_set():
    """dec_use_ifft=True forces the decoder dense regardless of
    attn_window; attn_window still masks the encoder."""
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=2, dec_use_ifft=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    assert ae.encoder is None and ae.encoder_masked is not None
    assert ae.decoder is not None and ae.decoder_masked is None and ae.decoder_fourier is None


def test_dec_use_ifft_shapes():
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=2, dec_use_ifft=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 8)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_dec_use_ifft_false_by_default():
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=8, hidden=16, n_blocks=1)
    ae = KSAutoencoderFourierMLP(cfg)
    assert ae.decoder is not None
    assert ae.decoder.input_proj.in_features == cfg.d_latent + 2 * ae.dec_modes


def test_dec_use_ifft_decoder_input_dim_is_d_latent_plus_NX():
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=8, hidden=16, n_blocks=1, dec_use_ifft=True)
    ae = KSAutoencoderFourierMLP(cfg)
    assert ae.decoder.input_proj.in_features == cfg.d_latent + cfg.NX


def test_dec_use_ifft_gradient_flows_end_to_end():
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=8, hidden=16, n_blocks=1, dec_use_ifft=True)
    ae = KSAutoencoderFourierMLP(cfg)
    u = torch.randn(4, 32, requires_grad=True)
    u_hat, z = ae.forward(u)
    (u_hat.sum() + z.sum()).backward()
    assert torch.isfinite(u.grad).all()


def test_dec_use_ifft_works_under_bfloat16_autocast():
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=2, dec_use_ifft=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    u = torch.randn(3, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        z = ae.encode(u)
        u_hat = ae.decode(z)
    assert torch.isfinite(z).all()
    assert torch.isfinite(u_hat).all()


def test_dec_use_ifft_checkpoint_round_trip(tmp_path):
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=2, dec_use_ifft=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    ae.eval()
    u = torch.randn(4, 32)
    u_hat_before = ae.decode(ae.encode(u))

    path = tmp_path / "ae.pt"
    torch.save(
        {"ae_state_dict": ae.state_dict(), "ae_config": cfg, "encoder_kind": "fourier_mlp"},
        path,
    )
    loaded, loaded_cfg, ckpt = load_autoencoder_checkpoint(str(path))
    assert isinstance(loaded, KSAutoencoderFourierMLP)
    loaded.eval()
    u_hat_after = loaded.decode(loaded.encode(u))
    assert torch.allclose(u_hat_before, u_hat_after)


# ---- fourier_ifft_readout option: the Fourier-only sub-network's output
# is interpreted as frequency-domain coefficients and explicitly inverse-
# transformed (irfft) back to state space, instead of an unconstrained
# linear readout -- added 2026-09-03, user-directed: "apply an inverse
# fft to the frequency component output of the fourier mlp to map back to
# state space in the encoder and propagator and decoder". Highest
# priority of the decoder options; decoder always goes fully dense under
# this ("dense for the decoder"). The masked raw-value path (attn_window,
# encoder only) is untouched -- still additive/non-interacting. ----


def test_fourier_ifft_readout_masked_encoder_and_decoder_same_structure():
    """encoder and decoder BOTH get the two-network structure (masked
    raw-value path SUMMED with FourierIFFTBody) -- decoder falls back to
    reusing attn_window when dec_attn_window isn't given separately."""
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=3, fourier_ifft_readout=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    from ks_latent.models.propagator import FourierIFFTBody
    assert ae.encoder is None and ae.encoder_masked is not None
    assert isinstance(ae.encoder_fourier, FourierIFFTBody)
    assert ae.decoder is None and ae.decoder_masked is not None
    assert isinstance(ae.decoder_fourier, FourierIFFTBody)
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 8)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_fourier_ifft_readout_dec_attn_window_independent_of_attn_window():
    """dec_attn_window, when given, overrides the decoder's window
    independently of the encoder's attn_window (e.g. a much larger
    decoder window than the encoder's -- a large enough window makes the
    mask trivially all-True/fully dense)."""
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=1, dec_attn_window=1000,
        fourier_ifft_readout=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    assert ae.decoder_masked.cross.mask.all()
    assert not ae.encoder_masked.cross.mask.all()


def test_fourier_ifft_readout_no_masked_path_when_attn_window_none():
    """attn_window=None (no encoder masking, no dec_attn_window given
    either): both encoder and decoder are pure FourierIFFTBody with no
    masked path at all."""
    cfg = FourierMLPAutoencoderConfig(NX=32, d_latent=8, hidden=16, n_blocks=1, fourier_ifft_readout=True)
    ae = KSAutoencoderFourierMLP(cfg)
    assert ae.encoder_masked is None
    assert ae.decoder_masked is None
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 8)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_fourier_ifft_readout_overrides_dec_use_ifft():
    """fourier_ifft_readout takes priority over dec_use_ifft for the
    decoder when both are set."""
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=3,
        dec_use_ifft=True, fourier_ifft_readout=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    from ks_latent.models.propagator import FourierIFFTBody
    assert isinstance(ae.decoder_fourier, FourierIFFTBody)
    assert ae.decoder is None


def test_fourier_ifft_readout_gradient_flows_end_to_end():
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=3, fourier_ifft_readout=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    u = torch.randn(4, 32, requires_grad=True)
    u_hat, z = ae.forward(u)
    (u_hat.sum() + z.sum()).backward()
    assert torch.isfinite(u.grad).all()


def test_fourier_ifft_readout_works_under_bfloat16_autocast():
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=3, fourier_ifft_readout=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    u = torch.randn(3, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        z = ae.encode(u)
        u_hat = ae.decode(z)
    assert torch.isfinite(z).all()
    assert torch.isfinite(u_hat).all()


def test_fourier_ifft_readout_checkpoint_round_trip(tmp_path):
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=3, fourier_ifft_readout=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    ae.eval()
    u = torch.randn(4, 32)
    u_hat_before = ae.decode(ae.encode(u))

    path = tmp_path / "ae.pt"
    torch.save(
        {"ae_state_dict": ae.state_dict(), "ae_config": cfg, "encoder_kind": "fourier_mlp"},
        path,
    )
    loaded, loaded_cfg, ckpt = load_autoencoder_checkpoint(str(path))
    assert isinstance(loaded, KSAutoencoderFourierMLP)
    loaded.eval()
    u_hat_after = loaded.decode(loaded.encode(u))
    assert torch.allclose(u_hat_before, u_hat_after)


# ---- nonexpansive: spectral-normalized layers + damped residuals +
# ortho FFT + averaged (not summed) dual-path combination -- added
# 2026-09-04, user-directed: "would there be a way to constrain the
# fourier_mlp to be nonexpansive". ----


def test_nonexpansive_shapes_and_gradient_flow():
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=3, fourier_ifft_readout=True,
        nonexpansive=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    u = torch.randn(5, 32, requires_grad=True)
    z = ae.encode(u)
    assert z.shape == (5, 8)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)
    (u_hat.sum() + z.sum()).backward()
    assert torch.isfinite(u.grad).all()


def test_nonexpansive_encoder_jacobian_norm_much_smaller_than_unconstrained():
    """Direct empirical check of the actual claim: spectral normalization
    + damped residuals substantially reduce the encoder's local
    amplification, not just in theory."""
    from torch.func import jacrev

    torch.manual_seed(0)
    kwargs = dict(NX=32, d_latent=8, hidden=16, n_blocks=2, attn_window=3, fourier_ifft_readout=True)
    ae_plain = KSAutoencoderFourierMLP(FourierMLPAutoencoderConfig(**kwargs, nonexpansive=False))
    ae_nonexp = KSAutoencoderFourierMLP(FourierMLPAutoencoderConfig(**kwargs, nonexpansive=True))
    ae_plain.eval()
    ae_nonexp.eval()
    u = torch.randn(32)

    def make_enc_fn(ae):
        def enc_fn(u_row):
            return ae.encode(u_row.unsqueeze(0)).squeeze(0)
        return enc_fn

    J_plain = jacrev(make_enc_fn(ae_plain))(u)
    J_nonexp = jacrev(make_enc_fn(ae_nonexp))(u)
    norm_plain = torch.linalg.svdvals(J_plain)[0].item()
    norm_nonexp = torch.linalg.svdvals(J_nonexp)[0].item()
    assert norm_nonexp < norm_plain


def test_nonexpansive_masked_path_effective_weight_zero_outside_band():
    """nonexpansive doesn't disable the masking itself: the EFFECTIVE
    (forward-used) weight `linear.weight * mask` is still exactly zero
    outside the band, since forward() re-applies the mask every call
    regardless of nonexpansive. NOTE (found directly, 2026-09-04): the
    RAW underlying spectral_norm-parametrized weight's off-band entries
    do NOT stay at exactly zero gradient under nonexpansive (unlike the
    non-nonexpansive case) -- spectral norm's estimated top singular
    value depends on the WHOLE matrix, so gradient flows into every raw
    entry via that normalization constant even though the mask still
    zeroes their contribution to the forward output. The forward-time
    functional guarantee (masked positions never affect the output)
    holds regardless; only the "raw weight literally never drifts
    off-band" cosmetic property (relied on by `masked_mlp_warm_start`)
    does not."""
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=2, fourier_ifft_readout=True,
        nonexpansive=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    mask = ae.encoder_masked.cross.mask
    W_effective = ae.encoder_masked.cross.linear.weight * mask
    assert (W_effective[~mask.bool()] == 0).all()


def test_nonexpansive_works_under_bfloat16_autocast():
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=3, fourier_ifft_readout=True,
        nonexpansive=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    u = torch.randn(3, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        z = ae.encode(u)
        u_hat = ae.decode(z)
    assert torch.isfinite(z).all()
    assert torch.isfinite(u_hat).all()


def test_nonexpansive_checkpoint_round_trip(tmp_path):
    cfg = FourierMLPAutoencoderConfig(
        NX=32, d_latent=8, hidden=16, n_blocks=1, attn_window=3, fourier_ifft_readout=True,
        nonexpansive=True,
    )
    ae = KSAutoencoderFourierMLP(cfg)
    ae.eval()
    u = torch.randn(4, 32)
    u_hat_before = ae.decode(ae.encode(u))

    path = tmp_path / "ae.pt"
    torch.save(
        {"ae_state_dict": ae.state_dict(), "ae_config": cfg, "encoder_kind": "fourier_mlp"},
        path,
    )
    loaded, loaded_cfg, ckpt = load_autoencoder_checkpoint(str(path))
    assert isinstance(loaded, KSAutoencoderFourierMLP)
    loaded.eval()
    u_hat_after = loaded.decode(loaded.encode(u))
    assert torch.allclose(u_hat_before, u_hat_after)
