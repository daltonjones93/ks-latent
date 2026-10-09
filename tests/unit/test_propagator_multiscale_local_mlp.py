"""Tests for `backbone="multiscale_local_mlp"` (added 2026-09-28) --
`_MultiscaleLocalMLPDeltaBody`, three independent `_LocalMLPDeltaBody`
instances (fine/regional/global), see `AuxPropagatorConfig`'s own
docstring for the design rationale.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import AuxPropagatorConfig, PropagatorConfig
from ks_latent.models.propagator import AuxPropagator, LatentPropagator


def _cfg(cls=PropagatorConfig, **overrides):
    defaults = dict(
        d_latent=62, mode="markovian", backbone="multiscale_local_mlp",
        multiscale_fine_n_sites=16, multiscale_fine_channels=3, multiscale_fine_window=3,
        multiscale_regional_n_sites=4, multiscale_regional_channels=3, multiscale_regional_window=1,
        multiscale_global_channels=2, multiscale_token_d_model=32, multiscale_n_layers=2,
    )
    defaults.update(overrides)
    return cls(**defaults)


def test_shape_and_finite():
    cfg = _cfg()
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 62)
    out = prop.step_one(z)
    assert out.shape == (4, 62)
    assert torch.isfinite(out).all()


def test_identity_at_init():
    cfg = _cfg(zero_init=True)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 62)
    out = prop.step_one(z)
    assert torch.allclose(out, z, atol=1e-5)


def test_not_identity_when_zero_init_false():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 62)
    out = prop.step_one(z)
    assert not torch.allclose(out, z, atol=1e-5)


def test_gradient_flows_to_all_three_scale_bodies():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z = torch.randn(4, 62)
    out = prop.step_one(z)
    out.sum().backward()
    body = prop.body
    fine_grad = body.fine_body.token_embed.weight.grad
    regional_grad = body.regional_body.token_embed.weight.grad
    # global_body is a plain MLPDeltaBody (see its own docstring: a
    # single-site scale has no spatial neighbors to convolve over, and
    # this also sidesteps a real MPS backward-kernel bug found while
    # building this), so its input layer is `input_proj`, not `token_embed`.
    global_grad = body.global_body.input_proj.weight.grad
    assert fine_grad is not None and fine_grad.abs().sum() > 0
    assert regional_grad is not None and regional_grad.abs().sum() > 0
    assert global_grad is not None and global_grad.abs().sum() > 0


def test_scales_are_independent_no_cross_talk():
    """By design, this first version has NO cross-scale coupling in the
    dynamics -- perturbing only the fine block must leave the
    regional/global outputs completely unchanged. dropout=0 and .eval()
    are required here: PropagatorConfig defaults dropout=0.1, and two
    separate train-mode forward passes draw independent dropout masks,
    which would make outputs differ even with no real cross-talk."""
    cfg = _cfg(zero_init=False, dropout=0.0)
    prop = LatentPropagator(cfg).eval()
    torch.manual_seed(0)
    z = torch.randn(4, 62)
    z_perturbed = z.clone()
    z_perturbed[:, :48] += 10.0  # perturb only the fine block
    out = prop.step_one(z)
    out_perturbed = prop.step_one(z_perturbed)
    assert torch.allclose(out[:, 48:], out_perturbed[:, 48:], atol=1e-5)
    assert not torch.allclose(out[:, :48], out_perturbed[:, :48], atol=1e-5)


def test_aux_propagator_config_works_too():
    cfg = _cfg(cls=AuxPropagatorConfig, zero_init=False)
    prop = AuxPropagator(cfg)
    z = torch.randn(4, 62)
    out = prop.step_one(z)
    assert out.shape == (4, 62)
    assert torch.isfinite(out).all()


def test_wrong_d_latent_rejected():
    with pytest.raises(ValueError, match="multiscale_local_mlp"):
        _cfg(d_latent=99)
