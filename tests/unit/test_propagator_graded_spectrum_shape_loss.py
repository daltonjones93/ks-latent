"""Tests for `propagator_graded_spectrum_shape_loss` (added 2026-09-10,
Section 134) -- a per-RANK generalization of `propagator_spectrum_shape_loss`'s
two-group floor, built after a robust 200-sample per-rank measurement of
Section 85's own real propagator showed the true spectrum is a smooth
graded decline across all d ranks, not two flat plateaus.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ks_latent.config import PropagatorConfig
from ks_latent.models.propagator import LatentPropagator
from ks_latent.models.spectral_field import encode_to_spectrum
from ks_latent.training.losses import propagator_graded_spectrum_shape_loss


class _DiagonalScaleMap(nn.Module):
    """z -> diag(scales) @ z -- Jacobian is exactly diag(scales) everywhere,
    so its singular values (already descending if scales is) are known
    exactly -- lets every test assert an EXACT expected number."""

    def __init__(self, scales: torch.Tensor):
        super().__init__()
        self.scales = scales

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.scales


def test_matches_reference_exactly_gets_zero_loss():
    torch.manual_seed(0)
    z = torch.randn(6, 5)
    scales = torch.tensor([2.0, 1.5, 1.0, 0.7, 0.4])
    m = _DiagonalScaleMap(scales)
    loss = propagator_graded_spectrum_shape_loss(m, z, target_spectrum=scales.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_exceeding_reference_everywhere_gets_zero_loss():
    torch.manual_seed(0)
    z = torch.randn(6, 5)
    scales = torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0])
    target = torch.tensor([2.0, 1.5, 1.0, 0.7, 0.4])
    m = _DiagonalScaleMap(scales)
    loss = propagator_graded_spectrum_shape_loss(m, z, target_spectrum=target)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_under_reference_at_one_rank_penalized_by_exact_amount():
    """`scales` must already be descending -- `svdvals` re-sorts regardless
    of the diagonal's construction order, so a non-descending `scales`
    would silently test a different rank assignment than intended."""
    torch.manual_seed(0)
    z = torch.randn(6, 5)
    scales = torch.tensor([2.0, 1.5, 0.5, 0.4, 0.3])  # rank 2 is under its target of 1.0
    target = torch.tensor([2.0, 1.5, 1.0, 0.4, 0.3])
    m = _DiagonalScaleMap(scales)
    loss = propagator_graded_spectrum_shape_loss(m, z, target_spectrum=target)
    expected = (1.0 - 0.5) ** 2 / 5  # relu(target-sv)^2, mean over 5 ranks, only rank 2 nonzero
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_reproduces_two_group_failure_mode_correctly():
    """Sanity check against the exact Section-130-style failure this was
    built to catch: top ranks fine, tail collapsed -- unlike the two-group
    loss with a low contract_floor, THIS reference explicitly expects the
    tail to be well above near-zero (matching a real measured spectrum),
    so it must penalize a collapsed tail heavily."""
    torch.manual_seed(0)
    z = torch.randn(6, 5)
    scales = torch.tensor([2.0, 1.8, 0.05, 0.05, 0.05])  # tail collapsed
    target = torch.tensor([2.0, 1.8, 1.0, 0.9, 0.8])  # graded reference expects real contraction, not collapse
    m = _DiagonalScaleMap(scales)
    loss = propagator_graded_spectrum_shape_loss(m, z, target_spectrum=target)
    assert loss.item() > 0.0
    expected = ((1.0 - 0.05) ** 2 + (0.9 - 0.05) ** 2 + (0.8 - 0.05) ** 2) / 5
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_gradient_flows_to_input():
    torch.manual_seed(0)
    z = torch.randn(4, 6, requires_grad=True)
    target = torch.tensor([1.5, 1.3, 1.1, 0.9, 0.7, 0.5])

    class _Nonlinear(nn.Module):
        def forward(self, z):
            return z * 0.3 + 0.1 * z**3

    loss = propagator_graded_spectrum_shape_loss(_Nonlinear(), z, target_spectrum=target)
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
    target = torch.linspace(1.5, 0.5, steps=2 * K)
    loss = propagator_graded_spectrum_shape_loss(prop.step_one, z, target_spectrum=target)
    assert torch.isfinite(loss).all()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_wrong_length_target_spectrum_raises():
    z = torch.randn(4, 5)
    m = _DiagonalScaleMap(torch.ones(5))
    with pytest.raises(ValueError, match="target_spectrum"):
        propagator_graded_spectrum_shape_loss(m, z, target_spectrum=torch.ones(4))
