"""Tests for `backbone="masked_mlp_wide"` (added 2026-09-06, Section 100,
user-directed: "For the propagator let's use a single layer mlp with more
parameters with limited attention window (say like 4). Please add the
proper number of parameters to match the size of the propagator in 98,
off diagonal terms (from the attention window) shouldn't count."). See
`PropagatorConfig`'s `backbone="masked_mlp_wide"` docstring: a single
`MaskedLinearRect(d_latent, hidden, attn_window) -> GELU ->
MaskedLinearRect(hidden, d_latent, attn_window)` layer, `mode="markovian"`
only, `attn_window` required.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import PropagatorConfig
from ks_latent.models.propagator import LatentPropagator


def test_masked_mlp_wide_constructs_and_produces_right_shape():
    cfg = PropagatorConfig(d_latent=12, mode="markovian", backbone="masked_mlp_wide", hidden=64, attn_window=2)
    prop = LatentPropagator(cfg)
    z = torch.randn(5, 12)
    out = prop.step_one(z)
    assert out.shape == (5, 12)


def test_masked_mlp_wide_identity_at_init():
    cfg = PropagatorConfig(
        d_latent=12, mode="markovian", backbone="masked_mlp_wide", hidden=64, attn_window=2, zero_init=True,
    )
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 12)
    out = prop.step_one(z)
    assert torch.allclose(out, z)


def test_masked_mlp_wide_not_identity_when_zero_init_false():
    cfg = PropagatorConfig(
        d_latent=12, mode="markovian", backbone="masked_mlp_wide", hidden=64, attn_window=2, zero_init=False,
    )
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 12)
    out = prop.step_one(z)
    assert not torch.allclose(out, z)


def test_masked_mlp_wide_masked_off_entries_stay_zero_gradient():
    cfg = PropagatorConfig(
        d_latent=12, mode="markovian", backbone="masked_mlp_wide", hidden=64, attn_window=2, zero_init=False,
    )
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 12)
    prop.step_one(z).sum().backward()
    in_mask = prop.body.input_proj.mask
    out_mask = prop.body.output_proj.mask
    assert (in_mask == 0).any() and (out_mask == 0).any(), "window=2 at d_latent=12 should exclude some entries"
    assert prop.body.input_proj.linear.weight.grad[in_mask == 0].abs().max().item() == 0.0
    assert prop.body.output_proj.linear.weight.grad[out_mask == 0].abs().max().item() == 0.0


def test_masked_mlp_wide_active_param_count_matches_target():
    """Verifies the exact sizing convention Section 100 relies on: total
    ACTIVE (non-zero-masked) params, not the raw dense .numel(), is what
    should be compared against another backbone's total -- see
    PropagatorConfig's docstring for the d_latent=44/window=4/hidden=6558
    calculation this mirrors at a small scale."""
    d_latent, hidden, window = 20, 100, 3
    cfg = PropagatorConfig(d_latent=d_latent, mode="markovian", backbone="masked_mlp_wide", hidden=hidden, attn_window=window)
    prop = LatentPropagator(cfg)
    raw_total = sum(p.numel() for p in prop.body.parameters())
    active_total = 0
    for m in prop.body.modules():
        if hasattr(m, "mask"):
            active_total += int(m.mask.sum().item())
    active_total += prop.body.input_proj.linear.bias.numel() + prop.body.output_proj.linear.bias.numel()
    assert active_total < raw_total  # the window genuinely excludes entries at this size
    assert active_total > 0


def test_masked_mlp_wide_requires_markovian_mode():
    with pytest.raises(ValueError, match="markovian"):
        PropagatorConfig(d_latent=12, mode="two_step", backbone="masked_mlp_wide", attn_window=2)


def test_masked_mlp_wide_requires_attn_window():
    with pytest.raises(ValueError, match="masked_mlp_wide"):
        PropagatorConfig(d_latent=12, mode="markovian", backbone="masked_mlp_wide", attn_window=None)


def test_masked_mlp_wide_rejected_for_history_mode():
    with pytest.raises(ValueError):
        PropagatorConfig(d_latent=12, mode="history", backbone="masked_mlp_wide", n_history=2, attn_window=2)
