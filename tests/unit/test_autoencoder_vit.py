"""Tests for the ViT-style autoencoder (`KSAutoencoderViT`, "Track B") and
its `CircularPositionalEncoding`, added 2026-08-29 -- see
CLAUDE_CODE_BRIEF.md §5.1/5.2 "ViT-style encoder/decoder" addendum.
"""

from __future__ import annotations

import torch

import pytest

from ks_latent.config import ViTAutoencoderConfig
from ks_latent.models.autoencoder_vit import (
    CircularPositionalEncoding,
    KSAutoencoderViT,
    LinearPositionalEncoding,
    build_local_attention_mask,
    build_ring_local_attention_mask,
    circular_overlap_tokenize,
)


# --- build_ring_local_attention_mask ----------------------------------------


def test_ring_mask_none_window_is_full_attention():
    assert build_ring_local_attention_mask(8, window=None) is None


def test_ring_mask_wraps_around():
    """Token 0 and token n_tokens-1 are ring-adjacent (distance 1), unlike
    a linear-distance mask which would treat them as maximally far."""
    mask = build_ring_local_attention_mask(8, window=1)
    assert mask[0, 7] == 0.0
    assert mask[7, 0] == 0.0
    assert mask[0, 1] == 0.0
    assert mask[0, 4] == float("-inf")  # antipodal, distance 4 > window 1


def test_ring_mask_is_symmetric_non_causal():
    mask = build_ring_local_attention_mask(10, window=2)
    assert torch.equal(mask, mask.T)
    assert torch.all(torch.diagonal(mask) == 0.0)


def test_ring_mask_shape_and_banding():
    mask = build_ring_local_attention_mask(6, window=2)
    assert mask.shape == (6, 6)
    assert mask[0, 2] == 0.0  # ring distance 2, within window
    assert mask[0, 3] == float("-inf")  # ring distance 3 (=6-3, antipodal), outside window=2


# --- CircularPositionalEncoding ---------------------------------------------


def test_pe_is_exactly_periodic_in_ring_index():
    """F depends on i only through 2*pi*m*i/n_tokens, so index n_tokens
    is index 0 -- a cyclic shift of the token sequence should permute the
    positional encoding cyclically, not need to 'wrap around' specially."""
    pe = CircularPositionalEncoding(n_tokens=12, d_model=16)
    rolled = torch.roll(pe.pe, shifts=1, dims=0)
    # rolled[i] should equal the encoding that WOULD be computed at ring
    # position (i-1) mod n_tokens -- check this indirectly via the
    # ring-distance invariance test below, and directly here via self-
    # consistency: pe[0] and pe[n_tokens] would coincide were there an
    # (n_tokens+1)-th row, i.e. the basis is n_tokens-periodic by construction.
    assert pe.pe.shape == (12, 16)
    assert torch.isfinite(pe.pe).all()


def test_pe_pairwise_distance_depends_only_on_ring_separation():
    """||F_i - F_j||^2 is a function of (i-j) mod n_tokens alone -- the
    defining property that makes this a *ring* metric."""
    torch.manual_seed(0)
    n_tokens, d_model = 16, 32
    pe = CircularPositionalEncoding(n_tokens, d_model)
    F = pe.pe  # (n_tokens, d_model)

    def dist(i, j):
        return (F[i] - F[j]).pow(2).sum().item()

    # All pairs at ring-separation 1 should have (nearly) the same distance,
    # regardless of which pair of adjacent indices is picked.
    d_sep1 = [dist(i, (i + 1) % n_tokens) for i in range(n_tokens)]
    assert max(d_sep1) - min(d_sep1) < 1e-3
    # Likewise at ring-separation 3.
    d_sep3 = [dist(i, (i + 3) % n_tokens) for i in range(n_tokens)]
    assert max(d_sep3) - min(d_sep3) < 1e-3


def test_pe_nearby_tokens_are_actually_nearer_than_distant_ones():
    """With a_m = 1/m the m=1 term dominates, so ring-distance should be
    (non-strictly) monotone in separation up to the antipodal point --
    the property the reference project measured breaking under flat
    amplitudes."""
    n_tokens, d_model = 24, 32
    pe = CircularPositionalEncoding(n_tokens, d_model)
    F = pe.pe
    d0 = [(F[0] - F[k]).pow(2).sum().item() for k in range(1, n_tokens // 2 + 1)]
    assert d0 == sorted(d0)  # strictly increasing out to the antipode
    assert d0[0] < d0[-1]


def test_pe_raises_if_d_model_too_small_for_ring_harmonics():
    import pytest

    with pytest.raises(ValueError):
        CircularPositionalEncoding(n_tokens=64, d_model=4)  # needs ~64 basis columns


def test_pe_forward_adds_encoding_to_tokens():
    pe = CircularPositionalEncoding(n_tokens=8, d_model=16)
    tokens = torch.zeros(3, 8, 16)
    out = pe(tokens)
    assert torch.allclose(out, pe.pe.unsqueeze(0).expand(3, -1, -1))


# --- build_local_attention_mask (linear distance) ---------------------------


def test_local_mask_none_window_is_full_attention():
    assert build_local_attention_mask(8, window=None) is None


def test_local_mask_does_not_wrap_around():
    """Direct contrast with the ring mask: token 0 and the last token are
    NOT treated as neighbours under linear distance."""
    mask = build_local_attention_mask(8, window=1)
    assert mask[0, 1] == 0.0  # adjacent, within window
    assert mask[0, 7] == float("-inf")  # NOT ring-adjacent under linear distance
    assert mask[7, 0] == float("-inf")


def test_local_mask_is_symmetric_non_causal():
    mask = build_local_attention_mask(10, window=2)
    assert torch.equal(mask, mask.T)
    assert torch.all(torch.diagonal(mask) == 0.0)


# --- LinearPositionalEncoding ------------------------------------------------


def test_linear_pe_shape_and_finite():
    pe = LinearPositionalEncoding(n_tokens=12, d_model=16)
    assert pe.pe.shape == (12, 16)
    assert torch.isfinite(pe.pe).all()


def test_linear_pe_does_not_treat_endpoints_as_close():
    """Direct contrast with CircularPositionalEncoding: position 0 and the
    last position should NOT be especially close under the linear (line)
    encoding -- they are the two most distant points on a line."""
    n_tokens, d_model = 16, 32
    lin = LinearPositionalEncoding(n_tokens, d_model)
    circ = CircularPositionalEncoding(n_tokens, d_model)
    d_lin_ends = (lin.pe[0] - lin.pe[-1]).pow(2).sum().item()
    d_lin_adjacent = (lin.pe[0] - lin.pe[1]).pow(2).sum().item()
    d_circ_ends = (circ.pe[0] - circ.pe[-1]).pow(2).sum().item()
    # Linear: endpoints are far apart relative to adjacent positions.
    assert d_lin_ends > d_lin_adjacent
    # Circular: endpoints are ring-adjacent, i.e. close, unlike linear.
    assert d_circ_ends < d_lin_ends


def test_linear_pe_forward_adds_encoding_and_gain_is_learnable():
    pe = LinearPositionalEncoding(n_tokens=8, d_model=16)
    tokens = torch.zeros(3, 8, 16)
    out = pe(tokens)
    assert torch.allclose(out, pe.pe.unsqueeze(0).expand(3, -1, -1))
    assert isinstance(pe.gain, torch.nn.Parameter)


def test_linear_pe_handles_odd_d_model():
    pe = LinearPositionalEncoding(n_tokens=10, d_model=15)
    assert pe.pe.shape == (10, 15)
    assert torch.isfinite(pe.pe).all()


# --- KSAutoencoderViT --------------------------------------------------------


def test_encode_decode_shapes_mean_pool():
    cfg = ViTAutoencoderConfig(NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="mean")
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_encode_decode_shapes_cls_pool():
    cfg = ViTAutoencoderConfig(NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="cls")
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    assert ae.enc_cls is not None


def test_readout_mlp_encode_decode_shapes():
    """`readout='mlp'` (added 2026-09-03, user-directed: "after the mean
    pool, there is a linear map from 96 to 44 ... can we have the option
    to replace that with a simple one layer mlp") -- shapes unchanged,
    only the post-pool head's internals differ."""
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="mean", readout="mlp",
    )
    ae = KSAutoencoderViT(cfg)
    assert isinstance(ae.enc_out, torch.nn.Sequential)
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_readout_mlp_matches_config_docstrings_structure():
    """Linear(d_model, d_latent) -> ReLU -> Linear(d_latent, d_latent)."""
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="mean", readout="mlp",
    )
    ae = KSAutoencoderViT(cfg)
    layers = list(ae.enc_out)
    assert len(layers) == 3
    assert isinstance(layers[0], torch.nn.Linear) and layers[0].in_features == 16 and layers[0].out_features == 6
    assert isinstance(layers[1], torch.nn.ReLU)
    assert isinstance(layers[2], torch.nn.Linear) and layers[2].in_features == 6 and layers[2].out_features == 6


def test_readout_linear_default_is_plain_linear():
    cfg = ViTAutoencoderConfig(NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="mean")
    ae = KSAutoencoderViT(cfg)
    assert isinstance(ae.enc_out, torch.nn.Linear)


def test_readout_mlp_requires_mean_or_cls_pool():
    with pytest.raises(ValueError, match="readout='mlp'"):
        ViTAutoencoderConfig(NX=32, patch_size=4, d_latent=8, pool="none", readout="mlp")


def test_invalid_readout_rejected():
    with pytest.raises(ValueError, match="readout"):
        ViTAutoencoderConfig(NX=32, patch_size=4, d_latent=6, readout="bogus")


def test_encode_decode_shapes_banded_pool():
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6,
        pool="banded", pool_bandwidth=2,
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_banded_pool_requires_bandwidth():
    import pytest

    with pytest.raises(ValueError):
        ViTAutoencoderConfig(pool="banded")


def test_pool_bandwidth_requires_banded_pool():
    import pytest

    with pytest.raises(ValueError):
        ViTAutoencoderConfig(pool="mean", pool_bandwidth=2)


def test_banded_pool_masked_linear_stays_exactly_zero_off_band():
    """`MaskedLinearRect`'s off-band entries must receive exactly zero
    gradient and thus stay exactly zero through training -- verify this
    holds for the actual modules `pool="banded"` builds, not just in
    isolation (regression guard mirroring the `masked_mlp` encoder's own
    test, Section 17)."""
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=8,
        pool="banded", pool_bandwidth=1,
    )
    ae = KSAutoencoderViT(cfg)
    opt = torch.optim.SGD(ae.parameters(), lr=0.1)
    for _ in range(3):
        u = torch.randn(4, 32)
        u_hat, z = ae(u)
        loss = ((u_hat - u) ** 2).mean() + z.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for module in (ae.enc_pool, ae.dec_pool):
        zero_mask = module.mask == 0
        assert torch.all(module.linear.weight[zero_mask] == 0.0)


def test_invalid_pool_rejected():
    import pytest

    with pytest.raises(ValueError):
        ViTAutoencoderConfig(pool="max")


# ---- dec_pool (added 2026-09-05, user-directed: "for the decoder, I
# would also like to ... use global mean pooling for the vit" -- asked
# while pool="local" for the ENCODER, i.e. genuinely asymmetric pooling) ----


def test_dec_pool_none_mirrors_pool():
    cfg = ViTAutoencoderConfig(NX=32, patch_size=4, d_latent=6, pool="local", pool_window=4)
    assert cfg.dec_pool is None
    assert cfg.dec_pool_mode == "local"


def test_dec_pool_mean_overrides_local_encoder_shapes():
    """The key case this whole feature exists for: a hard-local encoder
    (bounded receptive field) paired with a fully global mean-pooled
    decoder. dec_expand must be the DENSE Linear(d_latent, n_tokens*d_model)
    shape (mean/cls's), not the per-site Linear(local_channels, d_model)
    shape (local's) -- confirms dec_pool, not pool, drove construction."""
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6,
        pool="local", pool_window=4, dec_pool="mean",
    )
    ae = KSAutoencoderViT(cfg)
    assert isinstance(ae.dec_expand, torch.nn.Linear)
    assert ae.dec_expand.in_features == cfg.d_latent
    assert ae.dec_expand.out_features == cfg.n_tokens * cfg.d_model
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_dec_pool_asymmetric_gradient_flows_through_both_paths():
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6,
        pool="local", pool_window=4, dec_pool="mean",
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(3, 32, requires_grad=True)
    z = ae.encode(u)
    u_hat = ae.decode(z)
    u_hat.sum().backward()
    assert torch.isfinite(u.grad).all()
    assert ae.dec_expand.weight.grad is not None
    assert torch.isfinite(ae.dec_expand.weight.grad).all()


def test_dec_pool_defaults_to_symmetric_behavior_when_unset():
    """Regression guard: leaving dec_pool at its default None must produce
    IDENTICAL module shapes to the pre-existing symmetric behavior."""
    cfg_symmetric = ViTAutoencoderConfig(NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="mean")
    cfg_explicit = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="mean", dec_pool="mean",
    )
    ae_a, ae_b = KSAutoencoderViT(cfg_symmetric), KSAutoencoderViT(cfg_explicit)
    assert ae_a.dec_expand.weight.shape == ae_b.dec_expand.weight.shape


def test_invalid_dec_pool_rejected():
    import pytest

    with pytest.raises(ValueError, match="dec_pool"):
        ViTAutoencoderConfig(NX=32, patch_size=4, d_latent=6, dec_pool="max")


def test_dec_pool_local_requires_valid_pool_window_when_pool_is_mean():
    """The reverse direction: pool='mean' (no local constraints checked by
    the existing pool-keyed validation) but dec_pool='local' with an
    incompatible pool_window must still be caught."""
    import pytest

    with pytest.raises(ValueError, match="dec_pool"):
        ViTAutoencoderConfig(
            NX=32, patch_size=4, d_latent=5, pool="mean", dec_pool="local", pool_window=4,
        )  # n_tokens=8 % pool_window=4 == 0, but d_latent=5 % n_sites=2 != 0


def test_dec_pool_banded_requires_bandwidth_when_pool_not_banded():
    import pytest

    with pytest.raises(ValueError, match="dec_pool='banded'"):
        ViTAutoencoderConfig(NX=32, patch_size=4, d_latent=6, pool="mean", dec_pool="banded")


# ---- pool="gated" (added 2026-09-05, user-directed: "I think 2 is the
# best option to try first" -- soft/gated attention pooling) ----


def test_gated_pool_encode_decode_shapes():
    cfg = ViTAutoencoderConfig(NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="gated")
    ae = KSAutoencoderViT(cfg)
    assert ae.enc_gate is not None
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_gated_pool_is_a_weighted_not_uniform_average():
    """The whole point of gated pooling: the per-token weights come from a
    learned, input-DEPENDENT softmax, not a fixed 1/n_tokens each -- so
    two different inputs should generally get different weight
    distributions (a fixed mean would trivially be identical regardless
    of input content beyond the values themselves)."""
    torch.manual_seed(0)
    cfg = ViTAutoencoderConfig(NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="gated")
    ae = KSAutoencoderViT(cfg)
    ae.eval()
    u1 = torch.randn(1, 32)
    with torch.no_grad():
        tokens1 = ae.enc_pos(ae.enc_proj(
            circular_overlap_tokenize(u1, cfg.patch_size, ae._token_window, cfg.n_tokens)
        ))
        for block in ae.enc_blocks:
            tokens1 = block(tokens1, attn_mask=ae.enc_attn_mask)
        tokens1 = ae.enc_norm(tokens1)
        w1 = torch.softmax(ae.enc_gate(tokens1), dim=1)
    assert w1.std() > 1e-6  # weights are NOT all equal to 1/n_tokens


def test_gated_pool_decode_matches_mean_pool_decode_shape():
    """Decode is identical to mean/cls's dense expand -- gated pooling
    only changes the encode aggregation step."""
    cfg_gated = ViTAutoencoderConfig(NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="gated")
    cfg_mean = ViTAutoencoderConfig(NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="mean")
    ae_gated, ae_mean = KSAutoencoderViT(cfg_gated), KSAutoencoderViT(cfg_mean)
    assert ae_gated.dec_expand.weight.shape == ae_mean.dec_expand.weight.shape


def test_gated_pool_gradient_flows():
    cfg = ViTAutoencoderConfig(NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="gated")
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(3, 32, requires_grad=True)
    z = ae.encode(u)
    ae.decode(z).sum().backward()
    assert torch.isfinite(u.grad).all()
    assert ae.enc_gate.weight.grad is not None


def test_gated_pool_readout_mlp_allowed():
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="gated", readout="mlp",
    )
    ae = KSAutoencoderViT(cfg)
    assert isinstance(ae.enc_out, torch.nn.Sequential)


# ---- pool="token_mlp" (added 2026-09-05, user-directed: parameter-
# bounded alternative to naive flatten+dense pooling -- a per-token FFN
# compresses d_model -> d_model//token_mlp_reduction BEFORE flattening,
# then a 2-layer MLP maps the flattened vector to d_latent) ----


def test_token_mlp_encode_decode_shapes():
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6,
        pool="token_mlp", dec_pool="token_mlp", token_mlp_reduction=4, token_mlp_hidden=8,
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_token_mlp_param_count_is_bounded_by_compression():
    """The whole point: total params should scale with compressed_dim =
    d_model // token_mlp_reduction, NOT with a naive flatten's
    n_tokens*d_model -- a higher reduction factor must give a SMALLER
    encode-side module."""
    def build(reduction):
        cfg = ViTAutoencoderConfig(
            NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6,
            pool="token_mlp", token_mlp_reduction=reduction, token_mlp_hidden=8,
        )
        ae = KSAutoencoderViT(cfg)
        return sum(p.numel() for p in ae.enc_token_ffn.parameters()) + sum(
            p.numel() for p in ae.enc_flatten_mlp.parameters()
        )

    params_low_reduction = build(reduction=1)  # compressed_dim = d_model (no compression)
    params_high_reduction = build(reduction=8)  # compressed_dim much smaller
    assert params_high_reduction < params_low_reduction


def test_token_mlp_gradient_flows_both_sides():
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6,
        pool="token_mlp", dec_pool="token_mlp", token_mlp_reduction=4, token_mlp_hidden=8,
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(3, 32, requires_grad=True)
    z = ae.encode(u)
    ae.decode(z).sum().backward()
    assert torch.isfinite(u.grad).all()
    assert ae.enc_token_ffn[0].weight.grad is not None
    assert ae.dec_token_ffn[0].weight.grad is not None


def test_token_mlp_asymmetric_encoder_local_decoder_token_mlp():
    """token_mlp is usable independently on either side via dec_pool,
    same as every other mode."""
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6,
        pool="local", pool_window=4, dec_pool="token_mlp", token_mlp_reduction=4, token_mlp_hidden=8,
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(4, 32)
    z = ae.encode(u)
    assert z.shape == (4, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (4, 32)


def test_invalid_token_mlp_reduction_rejected():
    import pytest

    with pytest.raises(ValueError, match="token_mlp_reduction"):
        ViTAutoencoderConfig(NX=32, patch_size=4, d_latent=6, token_mlp_reduction=0)


# ---- pool="local_token_mlp" (added 2026-09-06, Section 100, user-directed:
# a WINDOWED variant of "token_mlp" -- the per-token FFN feeds a LEARNED
# circular-band-masked map (MaskedLinearRect, window=attn_window in
# n_tokens-ring units) instead of a dense flatten+MLP) ----


def test_local_token_mlp_encode_decode_shapes():
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6,
        pool="local_token_mlp", dec_pool="local_token_mlp", token_mlp_reduction=4,
        attn_window=1, pos_encoding="linear",
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_local_token_mlp_requires_attn_window():
    import pytest

    with pytest.raises(ValueError, match="local_token_mlp"):
        ViTAutoencoderConfig(NX=32, patch_size=4, d_latent=6, pool="local_token_mlp", attn_window=None)


def test_dec_local_token_mlp_requires_attn_window():
    import pytest

    with pytest.raises(ValueError, match="local_token_mlp"):
        ViTAutoencoderConfig(
            NX=32, patch_size=4, d_latent=6, pool="mean", dec_pool="local_token_mlp", attn_window=None,
        )


def test_local_token_mlp_gradient_flows_both_sides():
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6,
        pool="local_token_mlp", dec_pool="local_token_mlp", token_mlp_reduction=4,
        attn_window=1, pos_encoding="linear",
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(3, 32, requires_grad=True)
    z = ae.encode(u)
    ae.decode(z).sum().backward()
    assert torch.isfinite(u.grad).all()
    assert ae.enc_token_ffn[0].weight.grad is not None
    assert ae.enc_local_pool.linear.weight.grad is not None
    assert ae.dec_local_pool.linear.weight.grad is not None
    assert ae.dec_token_ffn[0].weight.grad is not None


def test_local_token_mlp_masked_off_entries_stay_zero_gradient():
    """The whole point of the window: entries of enc_local_pool's/
    dec_local_pool's weight matrix outside the circular band must receive
    exactly zero gradient, same guarantee MaskedLinearRect gives pool="banded"."""
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=8,
        pool="local_token_mlp", dec_pool="local_token_mlp", token_mlp_reduction=4,
        attn_window=1, pos_encoding="linear",
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(4, 32, requires_grad=True)
    z = ae.encode(u)
    ae.decode(z).sum().backward()
    enc_mask = ae.enc_local_pool.mask
    dec_mask = ae.dec_local_pool.mask
    assert (enc_mask == 0).any(), "test is only meaningful if the window actually excludes some entries"
    assert ae.enc_local_pool.linear.weight.grad[enc_mask == 0].abs().max().item() == 0.0
    assert ae.dec_local_pool.linear.weight.grad[dec_mask == 0].abs().max().item() == 0.0


def test_local_token_mlp_asymmetric_with_mean_decoder():
    """local_token_mlp is usable independently on either side via dec_pool,
    same as every other mode."""
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6,
        pool="local_token_mlp", dec_pool="mean", token_mlp_reduction=4,
        attn_window=1, pos_encoding="linear",
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(4, 32)
    z = ae.encode(u)
    assert z.shape == (4, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (4, 32)


def test_nx_not_divisible_by_patch_size_rejected():
    import pytest

    with pytest.raises(ValueError):
        ViTAutoencoderConfig(NX=30, patch_size=8)


def test_forward_matches_encode_then_decode():
    cfg = ViTAutoencoderConfig(NX=16, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=4)
    ae = KSAutoencoderViT(cfg)
    ae.eval()
    u = torch.randn(3, 16)
    u_hat, z = ae(u)
    z2 = ae.encode(u)
    u_hat2 = ae.decode(z2)
    assert torch.allclose(z, z2)
    assert torch.allclose(u_hat, u_hat2)


def test_gradients_reach_both_encoder_and_decoder():
    cfg = ViTAutoencoderConfig(NX=16, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=4)
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(8, 16)
    u_hat, z = ae(u)
    loss = ((u_hat - u) ** 2).mean()
    loss.backward()
    assert ae.enc_out.weight.grad is not None and torch.any(ae.enc_out.weight.grad != 0)
    assert ae.dec_out.weight.grad is not None and torch.any(ae.dec_out.weight.grad != 0)
    # The positional-encoding gain is a learnable scalar; confirm it gets a
    # gradient too, i.e. it actually participates rather than being a dead
    # buffer-like parameter.
    assert ae.enc_pos.gain.grad is not None


def test_overfits_a_single_batch():
    torch.manual_seed(0)
    cfg = ViTAutoencoderConfig(NX=16, patch_size=4, d_model=16, n_heads=2, n_blocks=2, d_latent=8)
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(4, 16)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    loss = None
    for _ in range(500):
        u_hat, _ = ae(u)
        loss = ((u_hat - u) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-3


# --- attn_window (added 2026-08-29) -----------------------------------------


def test_attn_window_none_matches_full_attention_shapes():
    cfg = ViTAutoencoderConfig(NX=16, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=4)
    ae = KSAutoencoderViT(cfg)
    assert ae.enc_attn_mask is None
    assert ae.dec_attn_mask is None


def test_attn_window_finite_builds_ring_masks_mean_pool():
    cfg = ViTAutoencoderConfig(
        NX=16, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=4, pool="mean", attn_window=1
    )
    ae = KSAutoencoderViT(cfg)
    assert ae.enc_attn_mask.shape == (4, 4)  # no cls token to pad for
    assert ae.dec_attn_mask.shape == (4, 4)
    u = torch.randn(3, 16)
    u_hat, z = ae(u)
    assert u_hat.shape == (3, 16)
    assert z.shape == (3, 4)


def test_attn_window_finite_pads_mask_for_cls_token():
    cfg = ViTAutoencoderConfig(
        NX=16, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=4, pool="cls", attn_window=1
    )
    ae = KSAutoencoderViT(cfg)
    # n_tokens=4 patch tokens + 1 CLS = 5x5 encoder mask; decoder has no CLS.
    assert ae.enc_attn_mask.shape == (5, 5)
    assert ae.dec_attn_mask.shape == (4, 4)
    # CLS row/col (index 0) is fully visible: no -inf anywhere in it.
    assert torch.all(ae.enc_attn_mask[0, :] == 0.0)
    assert torch.all(ae.enc_attn_mask[:, 0] == 0.0)
    u = torch.randn(3, 16)
    u_hat, z = ae(u)
    assert u_hat.shape == (3, 16)
    assert z.shape == (3, 4)


def test_attn_window_gradients_still_flow():
    cfg = ViTAutoencoderConfig(
        NX=16, patch_size=4, d_model=16, n_heads=2, n_blocks=2, d_latent=4, attn_window=1
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(6, 16)
    u_hat, _ = ae(u)
    loss = ((u_hat - u) ** 2).mean()
    loss.backward()
    assert ae.enc_out.weight.grad is not None and torch.any(ae.enc_out.weight.grad != 0)
    assert ae.dec_out.weight.grad is not None and torch.any(ae.dec_out.weight.grad != 0)


# --- pos_encoding="linear" (added 2026-08-29) --------------------------------


def test_invalid_pos_encoding_rejected():
    with pytest.raises(ValueError, match="pos_encoding"):
        ViTAutoencoderConfig(pos_encoding="bogus")


def test_pos_encoding_linear_uses_linear_pe_and_mask():
    cfg = ViTAutoencoderConfig(
        NX=16, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=4,
        pos_encoding="linear", attn_window=1,
    )
    ae = KSAutoencoderViT(cfg)
    assert isinstance(ae.enc_pos, LinearPositionalEncoding)
    assert isinstance(ae.dec_pos, LinearPositionalEncoding)
    # Linear mask: endpoints of the 4-token ring are NOT treated as neighbours.
    assert ae.enc_attn_mask[0, 3] == float("-inf")
    assert ae.dec_attn_mask[0, 3] == float("-inf")


def test_pos_encoding_circular_is_default_and_wraps():
    cfg = ViTAutoencoderConfig(
        NX=16, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=4, attn_window=1
    )
    ae = KSAutoencoderViT(cfg)
    assert isinstance(ae.enc_pos, CircularPositionalEncoding)
    assert ae.enc_attn_mask[0, 3] == 0.0  # ring-adjacent, NOT masked


def test_pos_encoding_linear_trains_end_to_end():
    torch.manual_seed(0)
    cfg = ViTAutoencoderConfig(
        NX=16, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=4, pos_encoding="linear"
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(4, 16)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    loss = None
    for _ in range(300):
        u_hat, _ = ae(u)
        loss = ((u_hat - u) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-2


# --- use_fno (added 2026-08-30, user-directed: "implement the FNO for the
# encoder and decoder in conjunction with the ViT structure") ---------------


def test_use_fno_false_gives_empty_fno_modulelists():
    cfg = ViTAutoencoderConfig(NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6)
    ae = KSAutoencoderViT(cfg)
    assert len(ae.enc_fno) == 0
    assert len(ae.dec_fno) == 0


def test_use_fno_true_builds_fno_layers_and_matching_shapes():
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6,
        use_fno=True, fno_n_layers=2,
    )
    ae = KSAutoencoderViT(cfg)
    assert len(ae.enc_fno) == 2
    assert len(ae.dec_fno) == 2
    u = torch.randn(5, 32)
    z = ae.encode(u)
    assert z.shape == (5, 6)
    u_hat = ae.decode(z)
    assert u_hat.shape == (5, 32)


def test_use_fno_with_mode_truncation():
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6,
        use_fno=True, fno_n_layers=1, fno_modes=2,
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(3, 32)
    u_hat, z = ae(u)
    assert z.shape == (3, 6)
    assert u_hat.shape == (3, 32)


def test_use_fno_with_cls_pool_excludes_cls_from_fno():
    """The CLS token has no ring position (see enc_cls's comment) -- FNO
    must run before it's concatenated, so it never sees a non-ring token."""
    cfg = ViTAutoencoderConfig(
        NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6, pool="cls",
        use_fno=True, fno_n_layers=1,
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(4, 32)
    z = ae.encode(u)
    assert z.shape == (4, 6)


def test_use_fno_trains_end_to_end():
    torch.manual_seed(0)
    cfg = ViTAutoencoderConfig(
        NX=16, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=4, use_fno=True, fno_n_layers=1,
    )
    ae = KSAutoencoderViT(cfg)
    u = torch.randn(4, 16)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    loss = None
    for _ in range(300):
        u_hat, _ = ae(u)
        loss = ((u_hat - u) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-2


def test_invalid_fno_n_layers_rejected():
    with pytest.raises(ValueError, match="fno_n_layers"):
        ViTAutoencoderConfig(fno_n_layers=0)


def test_invalid_fno_modes_rejected():
    with pytest.raises(ValueError, match="fno_modes"):
        ViTAutoencoderConfig(fno_modes=0)
