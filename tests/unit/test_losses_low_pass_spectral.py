"""Tests for `low_pass_spectral_loss` (added 2026-09-06, see
docs/sine_transform_pde_plan.md, `encoder_kind="spectral_field"` only).
"""

from __future__ import annotations

import math

import torch

from ks_latent.training.losses import low_pass_spectral_loss


def test_zero_z_gives_zero_loss():
    z = torch.zeros(3, 16)  # K=8
    loss = low_pass_spectral_loss(z, K=8, L=100.0, power=1.0)
    assert loss.item() == 0.0


def test_energy_only_in_dc_mode_gives_zero_loss():
    """k=0 at mode 0 (DC) -- any energy purely in that mode should
    contribute exactly 0 regardless of magnitude, since the weight there is
    0^power = 0."""
    K = 8
    z = torch.zeros(2, 2 * K)
    z[:, 0] = 5.0  # real part of mode 0
    loss = low_pass_spectral_loss(z, K, L=100.0, power=1.0)
    assert loss.item() == 0.0


def test_higher_mode_energy_contributes_more_than_lower_mode():
    K = 8
    z_low = torch.zeros(1, 2 * K)
    z_low[:, 1] = 1.0  # mode 1
    z_high = torch.zeros(1, 2 * K)
    z_high[:, 4] = 1.0  # mode 4, same magnitude, higher frequency
    loss_low = low_pass_spectral_loss(z_low, K, L=100.0, power=1.0)
    loss_high = low_pass_spectral_loss(z_high, K, L=100.0, power=1.0)
    assert loss_high.item() > loss_low.item()


def test_matches_hand_computed_value():
    K = 4
    L = 100.0
    z = torch.zeros(1, 2 * K)
    z[0, 2] = 3.0  # real part, mode 2
    z[0, 4 + 1] = 4.0  # imaginary part, mode 1
    loss = low_pass_spectral_loss(z, K, L, power=2.0)
    k2 = 2 * math.pi * 2 / L
    k1 = 2 * math.pi * 1 / L
    expected = (k2**2) * (3.0**2) + (k1**2) * (4.0**2)
    assert abs(loss.item() - expected) < 1e-5


def test_power_two_is_a_steeper_highk_vs_lowk_ratio_than_power_one():
    """power=2 (classical H^1 Sobolev) should penalize high-frequency
    energy relative to low-frequency energy more steeply than power=1
    (literal "proportional to frequency") -- i.e. the ratio
    loss(high-mode-only)/loss(low-mode-only) grows with power, regardless
    of whether k itself is above or below 1 in these particular units."""
    K = 8
    z_low = torch.zeros(1, 2 * K)
    z_low[0, 1] = 1.0  # mode 1
    z_high = torch.zeros(1, 2 * K)
    z_high[0, 7] = 1.0  # mode 7 (highest kept)

    ratio_p1 = (
        low_pass_spectral_loss(z_high, K, L=100.0, power=1.0).item()
        / low_pass_spectral_loss(z_low, K, L=100.0, power=1.0).item()
    )
    ratio_p2 = (
        low_pass_spectral_loss(z_high, K, L=100.0, power=2.0).item()
        / low_pass_spectral_loss(z_low, K, L=100.0, power=2.0).item()
    )
    assert ratio_p2 > ratio_p1 > 1.0


def test_batch_dimension_averaged_not_summed():
    K = 4
    z_one = torch.zeros(1, 2 * K)
    z_one[0, 1] = 1.0
    z_two = torch.cat([z_one, z_one], dim=0)
    loss_one = low_pass_spectral_loss(z_one, K, L=100.0, power=1.0)
    loss_two = low_pass_spectral_loss(z_two, K, L=100.0, power=1.0)
    assert abs(loss_one.item() - loss_two.item()) < 1e-6


def test_gradient_flows():
    K = 8
    z = torch.randn(3, 2 * K, requires_grad=True)
    loss = low_pass_spectral_loss(z, K, L=100.0, power=1.0)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    # gradient w.r.t. mode 0 (index 0 and index K) should be exactly zero
    # (k=0 there, so the loss doesn't depend on those entries at all).
    assert z.grad[:, 0].abs().max().item() == 0.0
    assert z.grad[:, K].abs().max().item() == 0.0
