"""Tests for `reference_mode_energy`/`spectral_shape_floor_loss` (added
2026-09-07, see docs/sine_transform_pde_plan.md, user-directed: "I really
just want to come up with a regularizer to prevent the latent state from
collapsing"). A ONE-SIDED floor calibrated from real training data's own
low-K-mode energy spectrum: penalizes z's per-mode energy share falling
BELOW the real-data reference, never for exceeding it.
"""

from __future__ import annotations

import math

import torch

from ks_latent.training.losses import reference_mode_energy, spectral_shape_floor_loss


def test_reference_mode_energy_sums_to_one():
    torch.manual_seed(0)
    u = torch.randn(50, 64)
    p_ref = reference_mode_energy(u, K=8)
    assert p_ref.shape == (8,)
    assert abs(p_ref.sum().item() - 1.0) < 1e-5
    assert (p_ref >= 0).all()


def test_reference_mode_energy_matches_hand_computed_single_mode_signal():
    """A pure single-frequency signal should put ~all its (nonzero-mode)
    energy at that one mode."""
    NX, L = 64, 22.0
    x = torch.arange(NX).float() * L / NX
    m = 3
    u = torch.cos(2 * math.pi * m * x / L).unsqueeze(0).repeat(20, 1)
    p_ref = reference_mode_energy(u, K=8)
    assert p_ref[m].item() > 0.9  # dominant mode
    other_mass = p_ref.sum().item() - p_ref[m].item()
    assert other_mass < 0.1


def _z_from_proportions(p: torch.Tensor, total_energy: float = 1.0) -> torch.Tensor:
    """Build a valid z (B=1, 2K) whose per-mode |z_k|^2 exactly matches
    `p * total_energy` (real part only, imaginary part zero -- simplest
    valid construction for testing the loss function directly)."""
    K = p.shape[0]
    real = torch.sqrt(p * total_energy).unsqueeze(0)  # (1, K)
    imag = torch.zeros(1, K)
    return torch.cat([real, imag], dim=-1)


def test_zero_loss_when_z_exactly_matches_reference():
    p_ref = torch.tensor([0.5, 0.3, 0.2])
    z = _z_from_proportions(p_ref)
    loss = spectral_shape_floor_loss(z, K=3, p_ref=p_ref)
    assert loss.item() < 1e-8


def test_penalized_when_a_mode_falls_below_reference():
    p_ref = torch.tensor([0.5, 0.3, 0.2])
    p_z = torch.tensor([0.5, 0.1, 0.4])  # mode 1 well below its 0.3 floor
    z = _z_from_proportions(p_z)
    loss = spectral_shape_floor_loss(z, K=3, p_ref=p_ref)
    assert loss.item() > 0.0


def test_not_penalized_when_all_modes_meet_or_exceed_reference():
    """One-sided: a mode with MORE than its reference share (compensating
    for information folded in from discarded higher modes) must not be
    penalized -- shift mass from mode 2 (well above floor) without ever
    letting any mode fall below its own floor."""
    p_ref = torch.tensor([0.2, 0.2, 0.2])
    p_z = torch.tensor([0.2, 0.2, 0.6])  # every mode >= its own reference share
    z = _z_from_proportions(p_z)
    loss = spectral_shape_floor_loss(z, K=3, p_ref=p_ref)
    assert loss.item() < 1e-8


def test_gradient_flows_when_below_floor():
    p_ref = torch.tensor([0.34, 0.33, 0.33])
    p_below = torch.tensor([0.5, 0.25, 0.25])  # modes 1, 2 below their own floor
    z = _z_from_proportions(p_below).requires_grad_(True)
    loss = spectral_shape_floor_loss(z, K=3, p_ref=p_ref)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert loss.item() > 0.0  # modes 1, 2 are below their floor


def test_batch_dimension_uses_mean_energy():
    torch.manual_seed(1)
    p_ref = torch.tensor([0.5, 0.3, 0.2])
    z_one = _z_from_proportions(p_ref)
    z_two = torch.cat([z_one, z_one], dim=0)
    loss_one = spectral_shape_floor_loss(z_one, K=3, p_ref=p_ref)
    loss_two = spectral_shape_floor_loss(z_two, K=3, p_ref=p_ref)
    assert abs(loss_one.item() - loss_two.item()) < 1e-6
