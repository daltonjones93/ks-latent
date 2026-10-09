"""Tests for `backbone="local_mlp_2d_aniso"` (`_LocalMLPDeltaBody2DAniso`,
added 2026-09-29, user-directed: "sure, build it" -- the anisotropic
Kolmogorov-flow analog of `local_mlp_2d`). Mirrors
`test_propagator_local_mlp_2d.py`'s discipline, with the key difference
being the y-axis equivariance tests: exact at a forcing-period site shift,
and explicitly NOT exact for a generic (non-period-multiple) y-shift --
this is the intended anisotropic behavior, not a bug (ground rule 19).
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import AuxPropagatorConfig, PropagatorConfig
from ks_latent.models.propagator import AuxPropagator, LatentPropagator


def _cfg(cls=PropagatorConfig, **overrides):
    defaults = dict(
        d_latent=192, mode="markovian", backbone="local_mlp_2d_aniso", attn_window=2,
        local_mlp2d_aniso_n_sites_x=8, local_mlp2d_aniso_n_sites_y=8,
        local_mlp2d_aniso_local_channels=3, local_mlp2d_aniso_token_d_model=32,
        local_mlp2d_aniso_n_layers=2, local_mlp2d_aniso_forcing_n=4,
    )
    defaults.update(overrides)
    return cls(**defaults)


def test_shape_and_finite():
    cfg = _cfg()
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    assert out.shape == (4, 192)
    assert torch.isfinite(out).all()


def test_identity_at_init():
    cfg = _cfg(zero_init=True)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    assert torch.allclose(out, z, atol=1e-5)


def test_not_identity_when_zero_init_false():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    assert not torch.allclose(out, z, atol=1e-5)


def test_exact_equivariance_under_x_site_shift():
    torch.manual_seed(0)
    cfg = _cfg(zero_init=False, dropout=0.0)
    prop = LatentPropagator(cfg).eval()
    Px, Py, c = 8, 8, 3
    z = torch.randn(2, Px * Py * c)
    z_field = z.reshape(2, Px, Py, c)
    z_shifted = torch.roll(z_field, shifts=1, dims=1).reshape(2, Px * Py * c)
    with torch.no_grad():
        out = prop.step_one(z)
        out_shifted = prop.step_one(z_shifted)
    out_field = out.reshape(2, Px, Py, c)
    out_shifted_field = out_shifted.reshape(2, Px, Py, c)
    assert torch.allclose(out_shifted_field, torch.roll(out_field, shifts=1, dims=1), atol=1e-5)


def test_exact_equivariance_under_y_forcing_period_shift():
    torch.manual_seed(1)
    cfg = _cfg(zero_init=False, dropout=0.0)
    prop = LatentPropagator(cfg).eval()
    Px, Py, c = 8, 8, 3
    forcing_n = 4
    period_sites = Py // forcing_n  # = 2
    z = torch.randn(2, Px * Py * c)
    z_field = z.reshape(2, Px, Py, c)
    z_shifted = torch.roll(z_field, shifts=period_sites, dims=2).reshape(2, Px * Py * c)
    with torch.no_grad():
        out = prop.step_one(z)
        out_shifted = prop.step_one(z_shifted)
    out_field = out.reshape(2, Px, Py, c)
    out_shifted_field = out_shifted.reshape(2, Px, Py, c)
    assert torch.allclose(out_shifted_field, torch.roll(out_field, shifts=period_sites, dims=2), atol=1e-5)


def test_generic_y_shift_NOT_equivariant():
    """Ground rule 19: deliberately does NOT generalize across an arbitrary
    y-shift -- if this started passing, the phase embedding had silently
    stopped depending on y (a regression), not an improvement."""
    torch.manual_seed(2)
    cfg = _cfg(zero_init=False, dropout=0.0)
    prop = LatentPropagator(cfg).eval()
    Px, Py, c = 8, 8, 3
    forcing_n = 4
    period_sites = Py // forcing_n  # = 2
    shift_sites = 1
    assert shift_sites % period_sites != 0
    z = torch.randn(2, Px * Py * c)
    z_field = z.reshape(2, Px, Py, c)
    z_shifted = torch.roll(z_field, shifts=shift_sites, dims=2).reshape(2, Px * Py * c)
    with torch.no_grad():
        out = prop.step_one(z)
        out_shifted = prop.step_one(z_shifted)
    out_field = out.reshape(2, Px, Py, c)
    out_shifted_field = out_shifted.reshape(2, Px, Py, c)
    rolled = torch.roll(out_field, shifts=shift_sites, dims=2)
    assert not torch.allclose(rolled, out_shifted_field, atol=1e-3)


def test_gradient_flows():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    out.sum().backward()
    body = prop.body
    assert body.token_embed.weight.grad is not None and body.token_embed.weight.grad.abs().sum() > 0
    assert body.phase_embed.weight.grad is not None and body.phase_embed.weight.grad.abs().sum() > 0
    assert body.blocks[0].mix.weight.grad is not None and body.blocks[0].mix.weight.grad.abs().sum() > 0


def test_aux_propagator_config_works_too():
    cfg = _cfg(cls=AuxPropagatorConfig, zero_init=False)
    prop = AuxPropagator(cfg)
    z = torch.randn(4, 192)
    out = prop.step_one(z)
    assert out.shape == (4, 192)
    assert torch.isfinite(out).all()


def test_missing_attn_window_rejected():
    with pytest.raises(ValueError, match="local_mlp_2d_aniso"):
        _cfg(attn_window=None)


def test_wrong_d_latent_rejected():
    with pytest.raises(ValueError, match="local_mlp_2d_aniso"):
        _cfg(d_latent=99)


def test_rejects_n_sites_y_not_multiple_of_forcing_n():
    with pytest.raises(ValueError, match="multiple of"):
        _cfg(local_mlp2d_aniso_n_sites_y=8, local_mlp2d_aniso_forcing_n=3)


def test_rollout_multistep_finite():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z0 = torch.randn(4, 192)
    roll = prop.rollout(z0, z0, k=5)
    assert roll.shape == (4, 5, 192)
    assert torch.isfinite(roll).all()
