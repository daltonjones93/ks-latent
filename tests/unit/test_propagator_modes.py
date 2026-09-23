"""Tests for the propagator mode flag (brief §5.2 addendum, 2026-08-29):
`"two_step"` ((z_{n-1},z_n) -> z_{n+1}, original) vs `"markovian"`
(M(z_n) -> z_{n+1}, since the KS PDE is first order in time), each with
`"mlp"` or (markovian-only) `"transformer"`/`"vit"` backbones (`"vit"`
added 2026-08-29, user-directed: same tokenize/detokenize shape as
`"transformer"`, but reuses `KSAutoencoderViT`'s circular positional
encoding + pre-norm ViT block, with no pooling/bottleneck step).
"""

from __future__ import annotations

import torch
import pytest

from ks_latent.config import AuxPropagatorConfig, PropagatorConfig
from ks_latent.models.propagator import AuxPropagator, LatentPropagator, build_propagator


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        PropagatorConfig(mode="bogus")


def test_invalid_backbone_raises():
    with pytest.raises(ValueError, match="backbone must be"):
        PropagatorConfig(backbone="bogus")


def test_two_step_with_transformer_backbone_raises():
    with pytest.raises(ValueError, match="only implemented for mode='markovian'"):
        PropagatorConfig(mode="two_step", backbone="transformer")


def test_two_step_with_vit_backbone_raises():
    with pytest.raises(ValueError, match="only implemented for mode='markovian'"):
        PropagatorConfig(mode="two_step", backbone="vit")


def test_transformer_backbone_requires_divisible_d_latent():
    with pytest.raises(ValueError, match="must be divisible"):
        PropagatorConfig(mode="markovian", backbone="transformer", d_latent=10, n_tokens=3)


def test_vit_backbone_requires_divisible_d_latent():
    with pytest.raises(ValueError, match="must be divisible"):
        PropagatorConfig(mode="markovian", backbone="vit", d_latent=10, n_tokens=3)


def test_two_step_with_fno_vit_backbone_raises():
    with pytest.raises(ValueError, match="only implemented for mode='markovian'"):
        PropagatorConfig(mode="two_step", backbone="fno_vit")


def test_history_with_fno_vit_backbone_is_valid():
    """`fno_vit` + `history` was added 2026-08-30 (user-directed: combine
    the FNO+ViT hybrid with the multi-step history propagator) -- this used
    to raise (only 'vit' was allowed for mode='history'); now it must
    construct and produce the right shape."""
    cfg = PropagatorConfig(
        d_latent=8, mode="history", backbone="fno_vit", n_history=3, n_tokens=4,
        token_d_model=8, token_nhead=2, token_n_layers=1,
    )
    prop = LatentPropagator(cfg)
    assert prop.mode == "history"
    z_hist = torch.randn(5, 3, 8)
    z_next = prop.step_history(z_hist)
    assert z_next.shape == (5, 8)
    rollout = prop.rollout_history(z_hist, k=3)
    assert rollout.shape == (5, 3, 8)


def test_history_with_mlp_backbone_is_valid():
    """`mlp` + `history` (added 2026-08-30, user-directed: after a plain MLP
    markovian propagator recovered chaos where every attention-based one
    collapsed -- docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 9 -- test
    whether the same holds with the user's originally-requested
    `n_history=3` recipe). Generalizes 'two_step' (fixed n_history=2) to an
    arbitrary history length by flattening (n_history, d) into one vector
    for the same `_MLPDeltaBody`, no new class."""
    cfg = PropagatorConfig(d_latent=8, mode="history", backbone="mlp", n_history=3, hidden=16, n_blocks=1)
    prop = LatentPropagator(cfg)
    assert prop.mode == "history"
    z_hist = torch.randn(5, 3, 8)
    z_next = prop.step_history(z_hist)
    assert z_next.shape == (5, 8)
    rollout = prop.rollout_history(z_hist, k=3)
    assert rollout.shape == (5, 3, 8)


def test_mlp_history_is_identity_at_init():
    cfg = PropagatorConfig(
        d_latent=8, mode="history", backbone="mlp", n_history=3, hidden=16, n_blocks=1, zero_init=True
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z_hist = torch.randn(5, 3, 8)
    z_next = prop.step_history(z_hist)
    assert torch.allclose(z_next, z_hist[:, -1], atol=1e-6)


def test_history_with_bogus_backbone_raises():
    with pytest.raises(ValueError, match="only implemented with backbone in"):
        PropagatorConfig(mode="history", backbone="transformer")


def test_fno_vit_backbone_requires_divisible_d_latent():
    with pytest.raises(ValueError, match="must be divisible"):
        PropagatorConfig(mode="markovian", backbone="fno_vit", d_latent=10, n_tokens=3)


def test_fno_vit_requires_positive_fno_n_layers():
    with pytest.raises(ValueError, match="fno_n_layers"):
        PropagatorConfig(mode="markovian", backbone="fno_vit", fno_n_layers=0)


def test_fno_vit_requires_positive_fno_modes():
    with pytest.raises(ValueError, match="fno_modes"):
        PropagatorConfig(mode="markovian", backbone="fno_vit", fno_modes=0)


# ---- "local_mlp" backbone (user-directed 2026-08-30): local receptive
# field (attn_window radius), but no softmax anywhere -- isolates receptive
# field width from the attention-softmax question. See
# docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 10. ----


def test_local_mlp_requires_attn_window():
    with pytest.raises(ValueError, match="requires attn_window"):
        PropagatorConfig(mode="markovian", backbone="local_mlp", attn_window=None)


def test_local_mlp_backbone_requires_divisible_d_latent():
    with pytest.raises(ValueError, match="must be divisible"):
        PropagatorConfig(mode="markovian", backbone="local_mlp", d_latent=10, n_tokens=3, attn_window=1)


def test_two_step_with_local_mlp_backbone_raises():
    with pytest.raises(ValueError, match="only implemented for mode='markovian'"):
        PropagatorConfig(mode="two_step", backbone="local_mlp", attn_window=1)


def test_history_with_local_mlp_backbone_raises():
    with pytest.raises(ValueError, match="only implemented with backbone in"):
        PropagatorConfig(mode="history", backbone="local_mlp", attn_window=1)


def test_local_mlp_shapes_and_identity_at_init():
    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="local_mlp", n_tokens=4, token_d_model=8,
        token_n_layers=1, attn_window=1, zero_init=True,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z_prev = torch.randn(5, 8)
    z_curr = torch.randn(5, 8)
    z_next = prop.step(z_prev, z_curr)
    assert z_next.shape == (5, 8)
    assert torch.allclose(z_next, z_curr, atol=1e-6)
    rollout = prop.rollout(z_prev, z_curr, k=3)
    assert rollout.shape == (5, 3, 8)


def test_local_mlp_conv_receptive_field_is_bounded():
    """A perturbation to one token must not reach a token outside the
    conv's `2*window+1` kernel radius, confirming the mixer really is
    local (unlike a dense/global layer, which would spread everywhere)."""
    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="local_mlp", n_tokens=8, token_d_model=8,
        token_n_layers=1, attn_window=1, zero_init=False, dropout=0.0,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z = torch.zeros(1, 8, requires_grad=True)
    out = prop.step_one(z)
    grad = torch.autograd.grad(out[0, 0], z)[0][0]  # d out[token 0] / d z (all tokens)
    nonzero = (grad.abs() > 1e-8).nonzero().flatten().tolist()
    # window=1, n_tokens=8 (chunk_size=1): token 0 can only be reached by
    # tokens {7, 0, 1} (circular neighbours within radius 1).
    assert set(nonzero) <= {7, 0, 1}
    assert 0 in nonzero  # token 0 must at least depend on itself


@pytest.mark.parametrize(
    "mode,backbone",
    [
        ("two_step", "mlp"), ("markovian", "mlp"), ("markovian", "transformer"),
        ("markovian", "vit"), ("markovian", "fno_vit"),
    ],
)
def test_propagator_shapes(mode, backbone):
    cfg = PropagatorConfig(
        d_latent=8, hidden=16, n_blocks=1, dropout=0.0, mode=mode, backbone=backbone,
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1,
    )
    prop = LatentPropagator(cfg)
    assert prop.mode == mode
    z_prev = torch.randn(5, 8)
    z_curr = torch.randn(5, 8)
    z_next = prop.step(z_prev, z_curr)
    assert z_next.shape == (5, 8)
    rollout = prop.rollout(z_prev, z_curr, k=4)
    assert rollout.shape == (5, 4, 8)


@pytest.mark.parametrize(
    "mode,backbone",
    [
        ("two_step", "mlp"), ("markovian", "mlp"), ("markovian", "transformer"),
        ("markovian", "vit"), ("markovian", "fno_vit"),
    ],
)
def test_propagator_is_identity_at_init(mode, backbone):
    cfg = PropagatorConfig(
        d_latent=8, hidden=16, n_blocks=1, dropout=0.0, zero_init=True, mode=mode, backbone=backbone,
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z_prev = torch.randn(6, 8)
    z_curr = torch.randn(6, 8)
    z_next = prop.step(z_prev, z_curr)
    assert torch.allclose(z_next, z_curr, atol=1e-6)


def test_markovian_step_ignores_z_prev():
    cfg = PropagatorConfig(d_latent=6, hidden=16, n_blocks=1, dropout=0.0, mode="markovian",
                            backbone="mlp", zero_init=False)
    torch.manual_seed(0)
    prop = LatentPropagator(cfg)
    prop.eval()
    z_curr = torch.randn(4, 6)
    out_a = prop.step(torch.randn(4, 6), z_curr)
    out_b = prop.step(torch.randn(4, 6), z_curr)  # different z_prev, same z_curr
    assert torch.allclose(out_a, out_b)


def test_markovian_step_one_matches_step():
    cfg = PropagatorConfig(d_latent=6, hidden=16, n_blocks=1, dropout=0.0, mode="markovian",
                            backbone="mlp", zero_init=False)
    torch.manual_seed(0)
    prop = LatentPropagator(cfg)
    prop.eval()
    z = torch.randn(4, 6)
    assert torch.allclose(prop.step_one(z), prop.step(z, z))


def test_two_step_actually_uses_z_prev():
    cfg = PropagatorConfig(d_latent=6, hidden=16, n_blocks=1, dropout=0.0, mode="two_step", zero_init=False)
    torch.manual_seed(0)
    prop = LatentPropagator(cfg)
    prop.eval()
    z_curr = torch.randn(4, 6)
    out_a = prop.step(torch.randn(4, 6), z_curr)
    out_b = prop.step(torch.randn(4, 6), z_curr)
    assert not torch.allclose(out_a, out_b)


def test_transformer_local_attention_mask_is_banded():
    from ks_latent.models.propagator import _build_local_attention_mask

    mask = _build_local_attention_mask(6, window=1)
    assert mask.shape == (6, 6)
    assert mask[0, 1] == 0.0  # within window
    assert mask[0, 2] == float("-inf")  # outside window
    assert torch.all(torch.diagonal(mask) == 0.0)


def test_transformer_full_attention_when_window_none():
    from ks_latent.models.propagator import _build_local_attention_mask

    assert _build_local_attention_mask(6, window=None) is None


def test_build_propagator_factory_handles_both_config_types():
    prop_cfg = PropagatorConfig(d_latent=5, hidden=8, n_blocks=1)
    aux_cfg = AuxPropagatorConfig(d_latent=5, hidden=8, n_blocks=1)
    p1 = build_propagator(prop_cfg)
    p2 = build_propagator(aux_cfg)
    assert isinstance(p1, LatentPropagator)
    assert isinstance(p2, LatentPropagator)


def test_aux_propagator_is_a_latent_propagator_subclass():
    aux_cfg = AuxPropagatorConfig(d_latent=5, hidden=8, n_blocks=1, mode="markovian", backbone="mlp")
    aux = AuxPropagator(aux_cfg)
    assert isinstance(aux, LatentPropagator)
    assert aux.mode == "markovian"


# --- "vit" backbone (added 2026-08-29) --------------------------------------


def test_vit_backbone_uses_circular_positional_encoding():
    """The "vit" backbone reuses KSAutoencoderViT's CircularPositionalEncoding
    (not the transformer backbone's learned nn.Parameter)."""
    from ks_latent.models.autoencoder_vit import CircularPositionalEncoding
    from ks_latent.models.propagator import _ViTDeltaBody

    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="vit",
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1,
    )
    prop = LatentPropagator(cfg)
    assert isinstance(prop.body, _ViTDeltaBody)
    assert isinstance(prop.body.pos, CircularPositionalEncoding)


def test_vit_backbone_has_no_pooling_or_bottleneck_step():
    """Unlike KSAutoencoderViT, the token grid keeps its full shape through
    every block -- no attribute resembling a pool/mean/cls step, and the
    number of blocks equals token_n_layers exactly (one ViTBlock per
    layer, nothing else in between)."""
    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="vit",
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=3,
    )
    prop = LatentPropagator(cfg)
    assert len(prop.body.blocks) == 3
    assert not hasattr(prop.body, "pool")
    assert not hasattr(prop.body, "cls")


def test_vit_backbone_respects_local_attention_mask():
    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="vit",
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1, attn_window=1,
    )
    prop = LatentPropagator(cfg)
    assert prop.body.attn_mask is not None
    assert prop.body.attn_mask.shape == (4, 4)


def test_vit_backbone_local_attention_mask_wraps_around_the_ring():
    """Regression test for a real bug caught 2026-08-29: the vit backbone's
    CircularPositionalEncoding assumes a ring, so its attention mask must
    use ring (not linear) distance -- token 0 and token n_tokens-1 must be
    treated as adjacent, unlike _TransformerDeltaBody's linear-distance
    mask (test_transformer_local_attention_mask_is_banded)."""
    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="vit",
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1, attn_window=1,
    )
    prop = LatentPropagator(cfg)
    mask = prop.body.attn_mask
    assert mask[0, 3] == 0.0  # ring-adjacent (distance 1), NOT masked
    assert mask[0, 2] == float("-inf")  # antipodal on a 4-ring, outside window=1


def test_vit_backbone_gradients_reach_token_embed_and_pos_gain():
    cfg = PropagatorConfig(
        d_latent=8, hidden=16, mode="markovian", backbone="vit", zero_init=False,
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1,
    )
    torch.manual_seed(0)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 8, requires_grad=True)
    out = prop.step_one(z)
    out.sum().backward()
    assert prop.body.token_embed.weight.grad is not None
    assert torch.any(prop.body.token_embed.weight.grad != 0)
    assert prop.body.pos.gain.grad is not None


def test_vit_backbone_param_count_scales_with_mlp_ratio():
    base = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="vit", token_mlp_ratio=1,
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1,
    )
    wide = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="vit", token_mlp_ratio=4,
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1,
    )
    n_base = sum(p.numel() for p in LatentPropagator(base).parameters())
    n_wide = sum(p.numel() for p in LatentPropagator(wide).parameters())
    assert n_wide > n_base


# --- vit backbone pos_encoding="linear" (added 2026-08-29) ------------------


def test_invalid_pos_encoding_rejected():
    with pytest.raises(ValueError, match="pos_encoding"):
        PropagatorConfig(pos_encoding="bogus")


def test_vit_backbone_pos_encoding_linear_uses_linear_pe_and_mask():
    from ks_latent.models.autoencoder_vit import LinearPositionalEncoding

    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="vit", pos_encoding="linear",
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1, attn_window=1,
    )
    prop = LatentPropagator(cfg)
    assert isinstance(prop.body.pos, LinearPositionalEncoding)
    # Linear mask: does NOT wrap around, unlike the circular default.
    assert prop.body.attn_mask[0, 3] == float("-inf")


def test_vit_backbone_pos_encoding_circular_default_wraps():
    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="vit",
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1, attn_window=1,
    )
    prop = LatentPropagator(cfg)
    assert prop.body.attn_mask[0, 3] == 0.0  # ring-adjacent, NOT masked


def test_vit_backbone_pos_encoding_linear_shape_and_identity_at_init():
    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="vit", pos_encoding="linear", zero_init=True,
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z = torch.randn(5, 8)
    z_next = prop.step_one(z)
    assert z_next.shape == (5, 8)
    assert torch.allclose(z_next, z, atol=1e-6)


# ---- "fno_vit" backbone (user-directed 2026-08-30, "Solution 2" for the
# fixed-point collapse -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md
# Section 5): FNO spectral-conv layers followed by ViT attention blocks. ----


def test_spectral_conv1d_shape():
    from ks_latent.models.autoencoder_vit import SpectralConv1d

    conv = SpectralConv1d(in_channels=6, out_channels=6, modes=3)
    x = torch.randn(4, 6, 10)
    out = conv(x)
    assert out.shape == (4, 6, 10)


def test_spectral_conv1d_all_modes_is_linear_in_input():
    """With `modes` covering every rfft frequency, the spectral conv is a
    fixed linear map of `x` (no truncation discards anything) -- a cheap
    sanity check that the FFT/mode-multiply/iFFT round trip is implemented
    correctly (linearity + no accidental nonlinearity anywhere in the path)."""
    from ks_latent.models.autoencoder_vit import SpectralConv1d

    N = 8
    conv = SpectralConv1d(in_channels=3, out_channels=3, modes=N // 2 + 1)
    x1 = torch.randn(2, 3, N)
    x2 = torch.randn(2, 3, N)
    out_sum = conv(x1 + x2)
    out_1 = conv(x1)
    out_2 = conv(x2)
    torch.testing.assert_close(out_sum, out_1 + out_2, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("fno_modes", [None, 2])
def test_fno_vit_shapes_with_and_without_mode_truncation(fno_modes):
    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="fno_vit", n_tokens=4, token_d_model=8,
        token_nhead=2, token_n_layers=1, fno_modes=fno_modes, fno_n_layers=2,
    )
    prop = LatentPropagator(cfg)
    z_prev = torch.randn(5, 8)
    z_curr = torch.randn(5, 8)
    z_next = prop.step(z_prev, z_curr)
    assert z_next.shape == (5, 8)
    rollout = prop.rollout(z_prev, z_curr, k=3)
    assert rollout.shape == (5, 3, 8)


def test_fno_vit_markovian_is_identity_at_init():
    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="fno_vit", zero_init=True,
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z = torch.randn(5, 8)
    assert torch.allclose(prop.step_one(z), z, atol=1e-6)


def test_fno_vit_history_is_identity_at_init():
    cfg = PropagatorConfig(
        d_latent=8, mode="history", backbone="fno_vit", zero_init=True, n_history=3,
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z_hist = torch.randn(5, 3, 8)
    z_next = prop.step_history(z_hist)
    assert torch.allclose(z_next, z_hist[:, -1], atol=1e-6)


# ---- "node" backbone (user-directed 2026-08-31): dz/dt = f_theta(z), a
# translation-equivariant local circular-conv vector field, integrated via
# fixed-step RK4 -- the most literal "model the latent as a PDE" backbone.
# See docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 42/43. ----


def test_node_requires_attn_window():
    with pytest.raises(ValueError, match="requires attn_window"):
        PropagatorConfig(mode="markovian", backbone="node", attn_window=None)


def test_node_requires_positive_ode_substeps():
    with pytest.raises(ValueError, match="ode_substeps"):
        PropagatorConfig(mode="markovian", backbone="node", attn_window=1, ode_substeps=0)


def test_two_step_with_node_backbone_raises():
    with pytest.raises(ValueError, match="only implemented for mode='markovian'"):
        PropagatorConfig(mode="two_step", backbone="node", attn_window=1)


def test_history_with_node_backbone_raises():
    with pytest.raises(ValueError, match="only implemented with"):
        PropagatorConfig(mode="history", backbone="node", attn_window=1)


def test_node_shapes_and_identity_at_init():
    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="node", hidden=8, n_blocks=1,
        attn_window=1, ode_substeps=3, zero_init=True,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z_prev = torch.randn(5, 8)
    z_curr = torch.randn(5, 8)
    z_next = prop.step(z_prev, z_curr)
    assert z_next.shape == (5, 8)
    assert torch.allclose(z_next, z_curr, atol=1e-6)
    rollout = prop.rollout(z_prev, z_curr, k=3)
    assert rollout.shape == (5, 3, 8)


def test_node_vector_field_receptive_field_is_bounded():
    """A perturbation to one latent index must not reach an index outside
    the RK4-compounded reach of the vector field's own local kernel,
    confirming the field really is local (unlike a dense/global layer,
    which would spread everywhere). `n_blocks=0` (only the field's single
    `in_conv`, radius `window`, contributes -- `out_conv` is `1x1`) keeps
    the per-evaluation reach exactly `window`, so with `ode_substeps=1`
    the RK4 stages (k1..k4) compound to a reach of exactly `4*window`."""
    window = 1
    d_latent = 32  # comfortably > 2*(4*window) so the bound is non-trivial
    cfg = PropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="node", hidden=4, n_blocks=0,
        attn_window=window, ode_substeps=1, zero_init=False, dropout=0.0,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z = torch.zeros(1, d_latent, requires_grad=True)
    out = prop.step_one(z)
    grad = torch.autograd.grad(out[0, 0], z)[0][0]  # d out[index 0] / d z (all indices)
    nonzero = (grad.abs() > 1e-8).nonzero().flatten().tolist()
    reach = 4 * window
    allowed = {i % d_latent for i in range(-reach, reach + 1)}
    assert set(nonzero) <= allowed
    assert 0 in nonzero  # index 0 must at least depend on itself
    assert len(allowed) < d_latent  # sanity: the bound is actually non-trivial (not global)


def test_node_vector_field_is_translation_equivariant():
    """Unlike `masked_mlp` (per-position-independent masked weights),
    `node`'s conv-based vector field applies the SAME kernel at every
    index -- shifting the input should shift the output identically."""
    d_latent = 16
    cfg = PropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="node", hidden=4, n_blocks=1,
        attn_window=2, ode_substeps=2, zero_init=False, dropout=0.0,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z = torch.randn(1, d_latent)
    shift = 3
    z_shifted = torch.roll(z, shifts=shift, dims=1)
    out = prop.step_one(z)
    out_shifted = prop.step_one(z_shifted)
    assert torch.allclose(torch.roll(out, shifts=shift, dims=1), out_shifted, atol=1e-5)


# ---- "cnn" backbone (user-directed 2026-09-01): "the best features of the
# current MLP" (input_proj -> n_blocks residual blocks -> final LayerNorm ->
# output_proj, zero-init) with cross-index Linears replaced by circular
# Conv1d of kernel WIDTH cnn_kernel_size (not a radius, unlike attn_window).
# See docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 43/44. ----


def test_cnn_requires_positive_kernel_size():
    with pytest.raises(ValueError, match="cnn_kernel_size"):
        PropagatorConfig(mode="markovian", backbone="cnn", cnn_kernel_size=0)


def test_two_step_with_cnn_backbone_raises():
    with pytest.raises(ValueError, match="only implemented for mode='markovian'"):
        PropagatorConfig(mode="two_step", backbone="cnn")


def test_history_with_cnn_backbone_raises():
    with pytest.raises(ValueError, match="only implemented with"):
        PropagatorConfig(mode="history", backbone="cnn")


@pytest.mark.parametrize("kernel_size", [3, 16])  # odd and even kernel widths
def test_cnn_shapes_and_identity_at_init(kernel_size):
    cfg = PropagatorConfig(
        d_latent=20, mode="markovian", backbone="cnn", hidden=8, n_blocks=2,
        cnn_kernel_size=kernel_size, zero_init=True,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z_prev = torch.randn(5, 20)
    z_curr = torch.randn(5, 20)
    z_next = prop.step(z_prev, z_curr)
    assert z_next.shape == (5, 20)
    assert torch.allclose(z_next, z_curr, atol=1e-6)
    rollout = prop.rollout(z_prev, z_curr, k=3)
    assert rollout.shape == (5, 3, 20)


def test_cnn_receptive_field_is_bounded():
    """A perturbation to one latent index must not reach an index outside
    the conv stack's reach, confirming the CNN really is local (unlike a
    dense/global layer). `kernel_size=3` (odd, symmetric padding +-1 per
    conv), `n_blocks=1` (2 sequential convs per residual block, no
    additional mixing from input_proj/output_proj, which are per-position)
    -> total reach exactly +-2."""
    d_latent = 16  # comfortably > 2*2+1 so the bound is non-trivial
    cfg = PropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="cnn", hidden=4, n_blocks=1,
        cnn_kernel_size=3, zero_init=False, dropout=0.0,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z = torch.zeros(1, d_latent, requires_grad=True)
    out = prop.step_one(z)
    grad = torch.autograd.grad(out[0, 0], z)[0][0]  # d out[index 0] / d z (all indices)
    nonzero = (grad.abs() > 1e-8).nonzero().flatten().tolist()
    reach = 2
    allowed = {i % d_latent for i in range(-reach, reach + 1)}
    assert set(nonzero) <= allowed
    assert 0 in nonzero
    assert len(allowed) < d_latent  # sanity: the bound is actually non-trivial (not global)


def test_cnn_is_translation_equivariant():
    """Weight-shared circular Conv1d kernels -> shifting the input should
    shift the output identically, unlike masked_mlp's per-position
    weights. Uses the default (even) cnn_kernel_size=16 to also exercise
    the asymmetric same-length circular padding end-to-end."""
    d_latent = 32
    cfg = PropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="cnn", hidden=4, n_blocks=1,
        cnn_kernel_size=16, zero_init=False, dropout=0.0,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z = torch.randn(1, d_latent)
    shift = 5
    z_shifted = torch.roll(z, shifts=shift, dims=1)
    out = prop.step_one(z)
    out_shifted = prop.step_one(z_shifted)
    assert torch.allclose(torch.roll(out, shifts=shift, dims=1), out_shifted, atol=1e-5)


# ---- "masked_mlp" backbone (user-directed 2026-08-30): dimension-preserving
# residual MLP with an optional circular-band mask on every Linear -- "local"
# (finite window) and "full" (window=None) share identical parameter shapes,
# enabling a direct weight-space warm start from local to full. See
# docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 11. ----


def test_masked_linear_zeros_off_band_at_init_and_keeps_them_zero():
    from ks_latent.models.propagator import MaskedLinear

    torch.manual_seed(0)
    lin = MaskedLinear(dim=8, window=1)
    off_band = lin.mask == 0
    assert torch.all(lin.linear.weight[off_band] == 0.0)
    # Off-band entries receive exactly zero gradient, so they stay zero
    # after an optimizer step even with large loss/gradient elsewhere.
    x = torch.randn(4, 8, requires_grad=True)
    out = lin(x)
    loss = out.sum()
    loss.backward()
    assert torch.all(lin.linear.weight.grad[off_band] == 0.0)


def test_masked_linear_window_none_is_fully_dense():
    from ks_latent.models.propagator import MaskedLinear

    lin = MaskedLinear(dim=6, window=None)
    assert torch.all(lin.mask == 1.0)


def test_masked_mlp_shapes_and_identity_at_init():
    cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="masked_mlp", n_blocks=1, attn_window=2, zero_init=True
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z = torch.randn(5, 8)
    z_next = prop.step_one(z)
    assert z_next.shape == (5, 8)
    assert torch.allclose(z_next, z, atol=1e-6)


def test_masked_mlp_allows_attn_window_none():
    """Unlike 'local_mlp', 'masked_mlp' explicitly ALLOWS attn_window=None
    (the fully-dense "full mlp" warm-start target) -- must not raise."""
    cfg = PropagatorConfig(d_latent=8, mode="markovian", backbone="masked_mlp", n_blocks=1, attn_window=None)
    prop = LatentPropagator(cfg)
    z = torch.randn(3, 8)
    assert prop.step_one(z).shape == (3, 8)


def test_masked_mlp_local_receptive_field_is_bounded():
    # n_blocks=1 -> 4 stacked MaskedLinear layers (input_proj, fc1, fc2,
    # output_proj), each with radius window=1 -> total reach = 4*1 = 4.
    # d_latent=20 (>> 2*reach) keeps this non-degenerate (reach=4 would
    # cover an 8-token ring entirely, testing nothing).
    d_latent, window, n_blocks = 20, 1, 1
    reach = 4 * window * n_blocks  # input_proj + fc1 + fc2 + output_proj per block
    cfg = PropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="masked_mlp", n_blocks=n_blocks,
        attn_window=window, zero_init=False,
    )
    prop = LatentPropagator(cfg)
    prop.eval()
    z = torch.zeros(1, d_latent, requires_grad=True)
    out = prop.step_one(z)
    grad = torch.autograd.grad(out[0, 0], z)[0][0]
    nonzero = (grad.abs() > 1e-8).nonzero().flatten().tolist()
    expected_reachable = {i % d_latent for i in range(-reach, reach + 1)}
    assert set(nonzero) <= expected_reachable
    assert len(nonzero) < d_latent  # confirms this is a genuine (non-degenerate) restriction


def test_masked_mlp_warm_start_copies_weights_and_perturbs_only_off_band():
    from ks_latent.models.propagator import masked_mlp_warm_start

    torch.manual_seed(0)
    local_cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="masked_mlp", n_blocks=1, attn_window=1, zero_init=False
    )
    local_prop = LatentPropagator(local_cfg)
    full_cfg = PropagatorConfig(
        d_latent=8, mode="markovian", backbone="masked_mlp", n_blocks=1, attn_window=None, zero_init=False
    )
    full_prop = masked_mlp_warm_start(local_prop, full_cfg, perturb_std=0.5, seed=1)

    local_w = local_prop.body.input_proj.linear.weight
    full_w = full_prop.body.input_proj.linear.weight
    mask = local_prop.body.input_proj.mask
    # On-band entries copied exactly.
    torch.testing.assert_close(full_w[mask == 1], local_w[mask == 1])
    # Off-band entries were exactly zero in the local model, now perturbed
    # (nonzero with overwhelming probability at perturb_std=0.5).
    assert torch.all(local_w[mask == 0] == 0.0)
    assert not torch.allclose(full_w[mask == 0], torch.zeros_like(full_w[mask == 0]))


def test_masked_mlp_warm_start_requires_local_then_full_configs():
    from ks_latent.models.propagator import masked_mlp_warm_start

    dense_cfg = PropagatorConfig(d_latent=8, mode="markovian", backbone="masked_mlp", attn_window=None)
    local_cfg = PropagatorConfig(d_latent=8, mode="markovian", backbone="masked_mlp", attn_window=1)
    dense_prop = LatentPropagator(dense_cfg)
    with pytest.raises(ValueError, match="finite attn_window"):
        masked_mlp_warm_start(dense_prop, dense_cfg)  # local_prop must be the masked one
    local_prop = LatentPropagator(local_cfg)
    with pytest.raises(ValueError, match="attn_window=None"):
        masked_mlp_warm_start(local_prop, local_cfg)  # full_cfg must be dense
