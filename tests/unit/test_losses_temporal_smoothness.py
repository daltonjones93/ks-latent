"""Tests for `ks_latent.training.losses.temporal_smoothness_loss` (added
2026-09-04, user-directed: "is there a way to encode the test function
here analyze_latent_smoothness.py as a regularization term ... implement
that" -- the differentiable, training-time version of
`scripts/analyze_latent_smoothness.py`'s real-trajectory step-size/
curvature diagnostic, following the finding that Section 82's GIF-visible
"huge jumps" were confirmed quantitatively by that diagnostic and that
`w_spatial`/D7/D8 regularize a different axis (cross-sectional channel
correlation, not z_t vs. z_{t+1})."""

from __future__ import annotations

import torch

from ks_latent.training.losses import temporal_smoothness_loss


def test_zero_for_perfectly_constant_trajectory():
    """A window where every time step is the identical vector has zero
    step size AND zero curvature -- the global minimum."""
    z0 = torch.randn(5, 8)
    z_win = z0.unsqueeze(1).expand(5, 6, 8)
    loss = temporal_smoothness_loss(z_win)
    assert loss.item() == 0.0


def test_positive_for_random_jumps():
    torch.manual_seed(0)
    z_win = torch.randn(20, 5, 10)
    loss = temporal_smoothness_loss(z_win)
    assert loss.item() > 0.0


def test_curvature_term_contributes_zero_for_window_length_2():
    """window=2 (e.g. mode='markovian' with k_pred=1) has no third point
    to form a second difference -- the loss must reduce to the plain
    step-size term alone, not error or silently drop the step term too."""
    torch.manual_seed(1)
    z_win = torch.randn(10, 2, 6)
    loss = temporal_smoothness_loss(z_win, curvature_weight=1000.0)  # weight must be irrelevant
    d = z_win.shape[-1]
    expected_step = (z_win[:, 1] - z_win[:, 0]).pow(2).sum(-1).mean() / d
    assert torch.allclose(loss, expected_step)


def test_curvature_term_zero_for_perfectly_linear_trajectory():
    """Constant velocity (no change in step vector) has zero second
    difference regardless of curvature_weight, even though step size is
    nonzero."""
    torch.manual_seed(2)
    z0 = torch.randn(4, 1, 7)
    v = torch.randn(4, 1, 7)
    steps = torch.arange(5).view(1, 5, 1)
    z_win = z0 + steps * v  # perfectly linear in time
    loss_w0 = temporal_smoothness_loss(z_win, curvature_weight=0.0)
    loss_w5 = temporal_smoothness_loss(z_win, curvature_weight=5.0)
    assert loss_w0.item() > 0.0  # step term is nonzero
    assert torch.allclose(loss_w0, loss_w5)  # curvature contributes 0 either way


def test_curvature_weight_scales_curvature_contribution():
    """For a trajectory with genuine curvature, raising curvature_weight
    must strictly increase the total loss (the step term is unaffected)."""
    torch.manual_seed(3)
    z_win = torch.randn(8, 4, 5)  # generic random window has nonzero curvature
    loss_low = temporal_smoothness_loss(z_win, curvature_weight=0.5)
    loss_high = temporal_smoothness_loss(z_win, curvature_weight=2.0)
    assert loss_high.item() > loss_low.item()


def test_decreases_with_gradient_descent_toward_smoothness():
    torch.manual_seed(4)
    z_win = torch.randn(30, 5, 8, requires_grad=True)
    opt = torch.optim.Adam([z_win], lr=0.05)
    loss0 = temporal_smoothness_loss(z_win).item()
    for _ in range(50):
        opt.zero_grad()
        loss = temporal_smoothness_loss(z_win)
        loss.backward()
        opt.step()
    loss1 = temporal_smoothness_loss(z_win).item()
    assert loss1 < loss0


def test_handles_small_batch_without_nan():
    torch.manual_seed(5)
    z_win = torch.randn(3, 3, 4, requires_grad=True)
    loss = temporal_smoothness_loss(z_win)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z_win.grad).all()


def test_works_under_bfloat16_autocast():
    torch.manual_seed(6)
    z_win = torch.randn(6, 4, 5, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = temporal_smoothness_loss(z_win.to(torch.bfloat16))
    assert torch.isfinite(loss)
    loss.float().backward()
    assert torch.isfinite(z_win.grad).all()
