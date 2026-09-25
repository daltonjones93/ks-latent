"""Tests for `backbone="site_conv"` (added 2026-09-24, Section 222, user-
directed: "could we use a model like the encoder from 221 as a propagator?
I've never thought of trying that"). See `_SiteConvDeltaBody`'s docstring:
reuses `KSAutoencoderLocalField`'s own circular-Conv1d site-mixing design
(`enc_mix`/`dec_mix`/`dec_in`) as a dimension-preserving `z -> z` map on
the latent's native `(n_sites, local_channels)` field. `mode="markovian"`
only, `attn_window` (the site-mixing radius) required, `d_latent` must
equal `site_conv_n_sites * site_conv_local_channels`.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import PropagatorConfig
from ks_latent.models.propagator import LatentPropagator


def _cfg(**overrides):
    defaults = dict(
        d_latent=12, mode="markovian", backbone="site_conv", attn_window=1,
        site_conv_n_sites=4, site_conv_local_channels=3, site_conv_hidden=8, site_conv_n_layers=2,
    )
    defaults.update(overrides)
    return PropagatorConfig(**defaults)


def test_site_conv_constructs_and_produces_right_shape():
    prop = LatentPropagator(_cfg())
    z = torch.randn(5, 12)
    out = prop.step_one(z)
    assert out.shape == (5, 12)


def test_site_conv_identity_at_init():
    prop = LatentPropagator(_cfg(zero_init=True))
    z = torch.randn(4, 12)
    out = prop.step_one(z)
    assert torch.allclose(out, z)


def test_site_conv_not_identity_when_zero_init_false():
    prop = LatentPropagator(_cfg(zero_init=False))
    z = torch.randn(4, 12)
    out = prop.step_one(z)
    assert not torch.allclose(out, z)


def test_site_conv_proj_out_has_no_bias():
    """Deliberate fix (see _SiteConvDeltaBody's docstring): the final
    projection must have bias=False, mirroring KSAutoencoderLocalField.
    enc_out's own documented fix for uncontrolled mean drift."""
    prop = LatentPropagator(_cfg(zero_init=False))
    assert prop.body.proj_out.bias is None


def test_site_conv_gradient_flows_to_input():
    prop = LatentPropagator(_cfg(zero_init=False))
    z = torch.randn(3, 12, requires_grad=True)
    out = prop.step_one(z)
    out.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum().item() > 0.0


def test_site_conv_receptive_field_matches_radius_times_n_layers():
    """Effective receptive field of `n_layers` stacked radius-`r` circular
    convs is `n_layers * r` sites (like stacking `n_layers` conv kernels
    of radius `r`) -- verified directly via the propagator's own Jacobian,
    the same measurement method used to calibrate Section 221's
    masked_mlp_expand attn_window choice."""
    n_sites, local_channels, radius, n_layers = 10, 2, 1, 2
    d_latent = n_sites * local_channels
    prop = LatentPropagator(
        _cfg(
            d_latent=d_latent, attn_window=radius, site_conv_n_sites=n_sites,
            site_conv_local_channels=local_channels, site_conv_hidden=6, site_conv_n_layers=n_layers,
            zero_init=False,
        )
    )
    z0 = torch.randn(d_latent)

    def f(zz):
        return prop.body(zz.unsqueeze(0)).squeeze(0)

    jac = torch.autograd.functional.jacobian(f, z0)  # (d_latent, d_latent)
    affected_by_0 = jac[:, 0].abs() > 1e-6
    idx = torch.arange(d_latent)
    site = idx // local_channels
    site_dist = torch.minimum(site.abs(), n_sites - site.abs())
    max_site_dist = site_dist[affected_by_0].max().item()
    assert max_site_dist <= n_layers * radius
    assert max_site_dist >= 1  # sanity: not degenerately disconnected


def test_site_conv_respects_circular_wraparound():
    """A site near the ring's edge (site n_sites-1) must be able to affect
    site 0's output when the radius reaches across the wrap, confirming
    padding_mode='circular' is actually active (not zero-padding)."""
    n_sites, local_channels, radius = 6, 2, 1
    d_latent = n_sites * local_channels
    prop = LatentPropagator(
        _cfg(
            d_latent=d_latent, attn_window=radius, site_conv_n_sites=n_sites,
            site_conv_local_channels=local_channels, site_conv_hidden=6, site_conv_n_layers=1,
            zero_init=False,
        )
    )
    z0 = torch.randn(d_latent)

    def f(zz):
        return prop.body(zz.unsqueeze(0)).squeeze(0)

    jac = torch.autograd.functional.jacobian(f, z0)  # (d_latent, d_latent)
    # last site's channels -> first site's channel-0 output row
    last_site_cols = slice((n_sites - 1) * local_channels, n_sites * local_channels)
    assert jac[0, last_site_cols].abs().max().item() > 1e-6


def test_site_conv_requires_markovian_mode():
    with pytest.raises(ValueError, match="markovian"):
        _cfg(mode="two_step")


def test_site_conv_requires_attn_window():
    with pytest.raises(ValueError, match="site_conv"):
        _cfg(attn_window=None)


def test_site_conv_requires_d_latent_matches_sites_times_channels():
    with pytest.raises(ValueError, match="site_conv"):
        _cfg(d_latent=13)  # 13 != 4*3


def test_site_conv_rejected_for_history_mode():
    with pytest.raises(ValueError):
        _cfg(mode="history", n_history=2)
