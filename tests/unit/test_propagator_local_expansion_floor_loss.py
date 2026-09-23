"""Tests for `propagator_local_expansion_floor_loss` (added 2026-09-09, see
its own docstring in `ks_latent.training.losses` for the full motivation):
a differentiable, one-sided floor on the propagator's own per-sample step-
Jacobian spectral norm, built on the same `torch.func.vmap(jacrev(...))`
technique `ks_latent.analysis.diagnostics.propagator_step_jacobian_
spectral_norms` already uses post-hoc for Gate 4's D9 diagnostic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ks_latent.config import PropagatorConfig
from ks_latent.models.propagator import LatentPropagator
from ks_latent.models.spectral_field import encode_to_spectrum
from ks_latent.training.losses import propagator_local_expansion_floor_loss


class _ScaleMap(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.scale


def test_expanding_map_gets_zero_loss():
    torch.manual_seed(0)
    z = torch.randn(6, 5)
    m = _ScaleMap(2.0)  # top singular value = 2 > floor
    loss = propagator_local_expansion_floor_loss(m, z, floor=1.0)
    assert loss.item() == 0.0


def test_contracting_map_gets_penalized_by_exact_amount():
    torch.manual_seed(0)
    z = torch.randn(6, 5)
    m = _ScaleMap(0.1)  # top singular value = 0.1 < floor=1.0
    loss = propagator_local_expansion_floor_loss(m, z, floor=1.0)
    assert abs(loss.item() - (1.0 - 0.1) ** 2) < 1e-5


def test_exactly_at_floor_gets_zero_loss():
    torch.manual_seed(0)
    z = torch.randn(6, 5)
    m = _ScaleMap(1.0)
    loss = propagator_local_expansion_floor_loss(m, z, floor=1.0)
    assert loss.item() == 0.0


class _NonlinearMap(nn.Module):
    """A LINEAR map's Jacobian is constant everywhere (independent of the
    input value), so there's genuinely no gradient signal for a floor
    loss to give back into z for one -- this isn't a bug, a linear map
    can't become more/less locally expansive at different points. Use a
    genuinely nonlinear map to test real gradient flow (the realistic
    case -- the actual spectral_pde propagator below is nonlinear too)."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * 0.3 + 0.1 * z**3


def test_gradient_flows_back_to_z():
    torch.manual_seed(0)
    z = torch.randn(4, 5, requires_grad=True)
    m = _NonlinearMap()
    loss = propagator_local_expansion_floor_loss(m, z, floor=1.0)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_works_with_real_spectral_pde_propagator_step_one():
    torch.manual_seed(0)
    K, N_w, L = 8, 32, 22.0
    cfg = PropagatorConfig(
        d_latent=2 * K, hidden=16, n_blocks=1, mode="markovian", backbone="spectral_pde",
        spectral_K=K, spectral_N_w=N_w, spectral_L=L, spectral_max_order=4,
        spectral_integrator="etdrk4", spectral_physics_prior=True,
        spectral_field_kind="polynomial", spectral_poly_degree=2, zero_init=True,
    )
    prop = LatentPropagator(cfg)
    w = torch.randn(4, N_w)
    z = encode_to_spectrum(w, K).requires_grad_(True)
    loss = propagator_local_expansion_floor_loss(prop.step_one, z, floor=1.0)
    assert torch.isfinite(loss).all()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
