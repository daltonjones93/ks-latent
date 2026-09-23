"""Tests for `backbone="spectral_pde_raw"` (added 2026-09-08, see
docs/sine_transform_pde_plan.md and `_SpectralPDERawDeltaBody`'s docstring):
generalizes `backbone="spectral_pde"` to work on the raw latent `z` of ANY
encoder (`(B, d_latent)`, no assumed spectral structure), by taking z's OWN
truncated self-FFT (treating its index as a spatial coordinate on a
periodic ring of circumference `L`) before running the ordinary
`_SpectralPDEDeltaBody` machinery, then transforming back.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import AuxPropagatorConfig, PropagatorConfig
from ks_latent.models.propagator import AuxPropagator, LatentPropagator


def _cfg(**overrides):
    d_latent = overrides.pop("d_latent", 20)
    defaults = dict(
        d_latent=d_latent, mode="markovian", backbone="spectral_pde_raw",
        spectral_K=d_latent // 2 + 1, spectral_L=float(d_latent), spectral_max_order=4,
        hidden=32, n_blocks=1,
    )
    defaults.update(overrides)
    return PropagatorConfig(**defaults)


def test_constructs_and_produces_right_shape():
    prop = LatentPropagator(_cfg())
    z = torch.randn(5, 20)
    out = prop.step_one(z)
    assert out.shape == (5, 20)


def test_identity_at_init_euler():
    prop = LatentPropagator(_cfg(zero_init=True, spectral_integrator="euler"))
    z = torch.randn(4, 20)
    out = prop.step_one(z)
    assert torch.allclose(out, z, atol=1e-4)


def test_etdrk4_at_init_is_not_identity():
    """Same non-identity property as backbone="spectral_pde": zero_init
    still reduces to the exact linearized map exp(Lhat)*z_hat in the
    self-FFT space, not a no-op."""
    prop = LatentPropagator(_cfg(zero_init=True, spectral_integrator="etdrk4"))
    z = torch.randn(4, 20)
    out = prop.step_one(z)
    assert not torch.allclose(out, z, atol=1e-3)


def test_gradient_flows_to_z_and_params():
    prop = LatentPropagator(_cfg(zero_init=False, spectral_integrator="etdrk4"))
    z = torch.randn(3, 20, requires_grad=True)
    out = prop.step_one(z)
    out.sum().backward()
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in prop.parameters())


def test_works_with_arbitrary_d_latent_not_matching_2K():
    """The defining difference from backbone="spectral_pde": no
    d_latent==2*spectral_K constraint -- spectral_K is purely an internal
    self-FFT parameter."""
    prop = LatentPropagator(_cfg(d_latent=44, spectral_K=10, spectral_L=44.0))
    z = torch.randn(2, 44)
    out = prop.step_one(z)
    assert out.shape == (2, 44)


def test_no_truncation_default_K_equals_d_latent_half_plus_one():
    d_latent = 20
    K = d_latent // 2 + 1
    prop = LatentPropagator(_cfg(d_latent=d_latent, spectral_K=K, spectral_L=float(d_latent)))
    assert prop.body.K == K
    assert prop.body.inner.K == K


def test_rejects_physics_prior():
    with pytest.raises(ValueError, match="spectral_physics_prior is not defined"):
        _cfg(spectral_physics_prior=True)


def test_rejects_missing_spectral_K():
    with pytest.raises(ValueError, match="requires spectral_K"):
        PropagatorConfig(
            d_latent=20, mode="markovian", backbone="spectral_pde_raw",
            spectral_K=None, spectral_L=20.0,
        )


def test_rejects_K_out_of_range():
    with pytest.raises(ValueError, match="must be in \\[1, d_latent//2\\+1"):
        PropagatorConfig(
            d_latent=20, mode="markovian", backbone="spectral_pde_raw",
            spectral_K=50, spectral_L=20.0,
        )


def test_rejects_two_step_mode():
    with pytest.raises(ValueError, match="only implemented for mode='markovian'"):
        PropagatorConfig(d_latent=20, mode="two_step", backbone="spectral_pde_raw", spectral_K=11)


def test_aux_propagator_config_same_validation():
    with pytest.raises(ValueError, match="spectral_physics_prior is not defined"):
        AuxPropagatorConfig(
            d_latent=20, mode="markovian", backbone="spectral_pde_raw",
            spectral_K=11, spectral_L=20.0, spectral_physics_prior=True,
        )
    cfg = AuxPropagatorConfig(
        d_latent=20, mode="markovian", backbone="spectral_pde_raw",
        spectral_K=11, spectral_L=20.0, hidden=16, n_blocks=1,
    )
    prop = AuxPropagator(cfg)
    out = prop.step_one(torch.randn(2, 20))
    assert out.shape == (2, 20)
