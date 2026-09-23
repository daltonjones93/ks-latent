"""Tests for `backbone="masked_mlp_expand"` (added 2026-09-10, Section 128,
user-directed: "I want to use a masked mlp with three layers,
attention_window=3 for the propagator... in the middle layer of the mlp,
expand the dimension 3x, but in this expanded dimensions, still respect the
attention window (I guess it would be 9 in that case, so only a very
limited number of neighboring values interact.)"). See `PropagatorConfig`'s
`backbone="masked_mlp_expand"` docstring: three `MaskedLinearRect` layers
(`input_proj`: d_latent->hidden, `mid_proj`: hidden->hidden, `output_proj`:
hidden->d_latent), `hidden = masked_mlp_expand_factor * d_latent`, all
sharing `attn_window` referenced against `d_latent`'s own ring. `mode=
"markovian"` only, `attn_window` required.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import PropagatorConfig
from ks_latent.models.propagator import LatentPropagator


def test_masked_mlp_expand_constructs_and_produces_right_shape():
    cfg = PropagatorConfig(
        d_latent=12, mode="markovian", backbone="masked_mlp_expand",
        attn_window=3, masked_mlp_expand_factor=3,
    )
    prop = LatentPropagator(cfg)
    z = torch.randn(5, 12)
    out = prop.step_one(z)
    assert out.shape == (5, 12)


def test_masked_mlp_expand_hidden_width_matches_expand_factor():
    cfg = PropagatorConfig(
        d_latent=12, mode="markovian", backbone="masked_mlp_expand",
        attn_window=3, masked_mlp_expand_factor=3,
    )
    prop = LatentPropagator(cfg)
    assert prop.body.input_proj.linear.weight.shape == (36, 12)  # (hidden, d_latent)
    assert prop.body.mid_proj.linear.weight.shape == (36, 36)  # (hidden, hidden)
    assert prop.body.output_proj.linear.weight.shape == (12, 36)  # (d_latent, hidden)


def test_masked_mlp_expand_identity_at_init():
    cfg = PropagatorConfig(
        d_latent=12, mode="markovian", backbone="masked_mlp_expand",
        attn_window=3, masked_mlp_expand_factor=3, zero_init=True,
    )
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 12)
    out = prop.step_one(z)
    assert torch.allclose(out, z)


def test_masked_mlp_expand_not_identity_when_zero_init_false():
    cfg = PropagatorConfig(
        d_latent=12, mode="markovian", backbone="masked_mlp_expand",
        attn_window=3, masked_mlp_expand_factor=3, zero_init=False,
    )
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 12)
    out = prop.step_one(z)
    assert not torch.allclose(out, z)


def test_masked_mlp_expand_masked_off_entries_stay_zero_gradient():
    cfg = PropagatorConfig(
        d_latent=12, mode="markovian", backbone="masked_mlp_expand",
        attn_window=2, masked_mlp_expand_factor=3, zero_init=False,
    )
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 12)
    prop.step_one(z).sum().backward()
    for layer in (prop.body.input_proj, prop.body.mid_proj, prop.body.output_proj):
        mask = layer.mask
        assert (mask == 0).any(), "window=2 at d_latent=12 should exclude some entries"
        assert layer.linear.weight.grad[mask == 0].abs().max().item() == 0.0


def test_masked_mlp_expand_mid_proj_radius_matches_window_times_expand_factor():
    """The user's own "I guess it would be 9" arithmetic, verified exactly:
    for a d_latent=44/window=3/expand_factor=3 setup, mid_proj (hidden ->
    hidden, hidden=132) should give each output slot a one-sided neighbor
    radius of exactly window*expand_factor=9 hidden-index positions (out of
    a 132-wide ring) -- see _MaskedMLPExpandDeltaBody's docstring for the
    derivation this checks."""
    d_latent, window, expand_factor = 44, 3, 3
    cfg = PropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="masked_mlp_expand",
        attn_window=window, masked_mlp_expand_factor=expand_factor,
    )
    prop = LatentPropagator(cfg)
    hidden = expand_factor * d_latent
    mask = prop.body.mid_proj.mask  # (hidden, hidden)
    assert mask.shape == (hidden, hidden)
    expected_radius = window * expand_factor
    row0 = mask[0]
    connected = row0.nonzero().flatten()
    # circular distance from index 0 for each connected column
    dist = torch.minimum(connected, hidden - connected)
    assert dist.max().item() == expected_radius
    assert int(row0.sum().item()) == 2 * expected_radius + 1


def test_masked_mlp_expand_dense_when_window_covers_whole_ring():
    """A window large enough to cover the whole ring should leave every
    mask entry active (sanity check on the mask construction, not just the
    narrow-window case)."""
    d_latent = 8
    cfg = PropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="masked_mlp_expand",
        attn_window=d_latent, masked_mlp_expand_factor=2,
    )
    prop = LatentPropagator(cfg)
    assert bool((prop.body.mid_proj.mask == 1).all())
    assert bool((prop.body.input_proj.mask == 1).all())
    assert bool((prop.body.output_proj.mask == 1).all())


def test_masked_mlp_expand_gradient_flows_to_input():
    cfg = PropagatorConfig(
        d_latent=10, mode="markovian", backbone="masked_mlp_expand",
        attn_window=2, masked_mlp_expand_factor=3, zero_init=False,
    )
    prop = LatentPropagator(cfg)
    z = torch.randn(3, 10, requires_grad=True)
    out = prop.step_one(z)
    out.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum().item() > 0.0


def test_masked_mlp_expand_requires_markovian_mode():
    with pytest.raises(ValueError, match="markovian"):
        PropagatorConfig(d_latent=12, mode="two_step", backbone="masked_mlp_expand", attn_window=3)


def test_masked_mlp_expand_requires_attn_window():
    with pytest.raises(ValueError, match="masked_mlp_expand"):
        PropagatorConfig(d_latent=12, mode="markovian", backbone="masked_mlp_expand", attn_window=None)


def test_masked_mlp_expand_rejected_for_history_mode():
    with pytest.raises(ValueError):
        PropagatorConfig(
            d_latent=12, mode="history", backbone="masked_mlp_expand", n_history=2, attn_window=2,
        )


def test_masked_mlp_expand_factor_must_be_positive():
    with pytest.raises(ValueError, match="masked_mlp_expand_factor"):
        PropagatorConfig(
            d_latent=12, mode="markovian", backbone="masked_mlp_expand",
            attn_window=2, masked_mlp_expand_factor=0,
        )
