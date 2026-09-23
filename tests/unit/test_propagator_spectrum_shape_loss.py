"""Tests for `propagator_spectrum_shape_loss` (added 2026-09-10, Section
131, user-directed: "can we use the regularizer to force some singular
vectors to have expansive values around 1.5 and others to have contracting
values?" -- a direct fix for `propagator_local_expansion_floor_loss`'s own
diagnosed failure mode: constraining only the top singular value let the
other d-1 collapse toward zero (measured on a real Section 130 checkpoint:
only 5/44 singular values >= 1.0, per-step volume factor 8.9e-5). This
shapes the WHOLE spectrum: the top `n_expand` singular values get an
expansive floor, the rest get a separate (lower) contracting floor.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ks_latent.config import PropagatorConfig
from ks_latent.models.propagator import LatentPropagator
from ks_latent.models.spectral_field import encode_to_spectrum
from ks_latent.training.losses import propagator_spectrum_shape_loss


class _DiagonalScaleMap(nn.Module):
    """z -> diag(scales) @ z -- a LINEAR map with an exactly known,
    constant-everywhere singular value spectrum equal to `scales.abs()`,
    sorted descending. Lets every test below assert an EXACT expected
    number, not just a direction."""

    def __init__(self, scales: torch.Tensor):
        super().__init__()
        self.scales = scales

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.scales


def test_matches_target_gets_zero_loss():
    torch.manual_seed(0)
    z = torch.randn(6, 5)
    # 2 expansive (2.0), 3 contracting (0.5) -- both at exactly their targets.
    scales = torch.tensor([2.0, 2.0, 0.5, 0.5, 0.5])
    m = _DiagonalScaleMap(scales)
    loss = propagator_spectrum_shape_loss(
        m, z, n_expand=2, expand_target=2.0, contract_floor=0.5,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_under_expanded_top_is_penalized():
    torch.manual_seed(0)
    z = torch.randn(6, 5)
    # top group at 1.0, well under expand_target=1.5; contracting group
    # comfortably clears its own floor.
    scales = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5])
    m = _DiagonalScaleMap(scales)
    loss = propagator_spectrum_shape_loss(
        m, z, n_expand=2, expand_target=1.5, contract_floor=0.3,
    )
    expected = (1.5 - 1.0) ** 2  # relu(expand_target - sv)^2, uniform across top group
    assert loss.item() == pytest.approx(expected, abs=1e-4)


def test_collapsed_tail_is_penalized_even_when_top_is_fine():
    """The exact failure mode this loss was built to catch: top singular
    values clear their expansive floor, but the tail has collapsed toward
    zero -- propagator_local_expansion_floor_loss (top-1 only) would score
    this as PERFECT (loss=0); this function must not."""
    torch.manual_seed(0)
    z = torch.randn(6, 5)
    scales = torch.tensor([2.0, 2.0, 0.05, 0.05, 0.05])
    m = _DiagonalScaleMap(scales)
    loss = propagator_spectrum_shape_loss(
        m, z, n_expand=2, expand_target=1.5, contract_floor=0.7,
    )
    expected_contract = (0.7 - 0.05) ** 2
    assert loss.item() == pytest.approx(expected_contract, abs=1e-4)
    assert loss.item() > 0.0


def test_exceeding_targets_gets_zero_loss_one_sided():
    torch.manual_seed(0)
    z = torch.randn(6, 5)
    # top group WAY above expand_target, contracting group WAY above its floor too.
    scales = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.5])
    m = _DiagonalScaleMap(scales)
    loss = propagator_spectrum_shape_loss(
        m, z, n_expand=2, expand_target=1.5, contract_floor=0.5,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_gradient_flows_to_input():
    torch.manual_seed(0)
    z = torch.randn(4, 6, requires_grad=True)

    class _Nonlinear(nn.Module):
        def forward(self, z):
            return z * 0.3 + 0.1 * z**3

    loss = propagator_spectrum_shape_loss(_Nonlinear(), z, n_expand=2, expand_target=1.5, contract_floor=0.7)
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
    loss = propagator_spectrum_shape_loss(prop.step_one, z, n_expand=4, expand_target=1.0, contract_floor=0.5)
    assert torch.isfinite(loss).all()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_n_expand_zero_raises():
    z = torch.randn(4, 5)
    m = _DiagonalScaleMap(torch.ones(5))
    with pytest.raises(ValueError, match="n_expand"):
        propagator_spectrum_shape_loss(m, z, n_expand=0)


def test_n_expand_equal_to_d_raises():
    z = torch.randn(4, 5)
    m = _DiagonalScaleMap(torch.ones(5))
    with pytest.raises(ValueError, match="n_expand"):
        propagator_spectrum_shape_loss(m, z, n_expand=5)
