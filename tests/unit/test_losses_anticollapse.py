"""Tests for `ks_latent.training.losses.variance_floor_loss` and
`.logdet_barrier_loss` (added 2026-08-31, user-directed -- Phase 2
architecture doc Section 35/38 options 3a/3b: two new Stage-1 anti-collapse
regularizers, following the discovery that extended joint training can
collapse one latent channel's variance many orders of magnitude below the
rest, undetected by the existing `w_var`/`decorr_var_loss`)."""

from __future__ import annotations

import torch

from ks_latent.training.losses import logdet_barrier_loss, variance_floor_loss


# --- variance_floor_loss ---------------------------------------------------


def test_variance_floor_zero_when_all_channels_above_gamma():
    torch.manual_seed(0)
    z = torch.randn(500, 20) * 2.0  # every channel std ~2.0
    loss = variance_floor_loss(z, gamma=0.1)
    assert loss.item() == 0.0


def test_variance_floor_positive_when_a_channel_is_collapsed():
    torch.manual_seed(1)
    z = torch.randn(500, 20)
    z[:, 0] *= 1e-4  # collapse one channel far below gamma
    loss = variance_floor_loss(z, gamma=0.1)
    assert loss.item() > 0.0


def test_variance_floor_does_not_penalize_a_strong_channel():
    """The key property distinguishing this from the existing `w_var`
    (`decorr_var_loss`'s `l_var`, which pulls every channel toward
    variance exactly 1): a channel with std FAR ABOVE gamma pays nothing,
    however large it is."""
    torch.manual_seed(2)
    z = torch.randn(500, 10) * 10.0  # every channel std ~10, way above any sane gamma
    loss = variance_floor_loss(z, gamma=0.1)
    assert loss.item() == 0.0


def test_variance_floor_decreases_with_gradient_descent_on_collapsed_channel():
    torch.manual_seed(3)
    z = torch.randn(200, 8, requires_grad=True)
    with torch.no_grad():
        z[:, 0] *= 1e-3
    opt = torch.optim.Adam([z], lr=0.05)
    loss0 = variance_floor_loss(z, gamma=0.2).item()
    for _ in range(50):
        opt.zero_grad()
        loss = variance_floor_loss(z, gamma=0.2)
        loss.backward()
        opt.step()
    loss1 = variance_floor_loss(z, gamma=0.2).item()
    assert loss1 < loss0


def test_variance_floor_handles_small_batch_without_nan():
    torch.manual_seed(4)
    z = torch.randn(4, 6, requires_grad=True)
    loss = variance_floor_loss(z)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()


# --- logdet_barrier_loss ----------------------------------------------------


def test_logdet_barrier_higher_for_collapsed_than_healthy_spectrum():
    torch.manual_seed(5)
    d, n = 20, 2000
    z_healthy = torch.randn(n, d)
    z_collapsed = torch.randn(n, d)
    z_collapsed[:, 0] *= 1e-4  # one channel's variance collapsed
    loss_healthy = logdet_barrier_loss(z_healthy, eps=1e-6)
    loss_collapsed = logdet_barrier_loss(z_collapsed, eps=1e-6)
    assert loss_collapsed > loss_healthy


def test_logdet_barrier_reacts_to_correlation_driven_collapse_unlike_variance_floor():
    """The key property distinguishing this from `variance_floor_loss`:
    two channels that are each individually healthy-variance but nearly
    perfectly linearly dependent (a small JOINT eigenvalue, no single
    small marginal variance) should still score worse than independent
    channels under logdet_barrier_loss, even though variance_floor_loss
    can't see this at all (every marginal std is healthy)."""
    torch.manual_seed(6)
    n = 2000
    x = torch.randn(n, 1)
    z_redundant = torch.cat([x, x + 1e-4 * torch.randn(n, 1), torch.randn(n, 8)], dim=1)
    z_independent = torch.randn(n, 10)
    # Marginal per-channel variances are all healthy in both cases.
    assert variance_floor_loss(z_redundant, gamma=0.5).item() == 0.0
    assert variance_floor_loss(z_independent, gamma=0.5).item() == 0.0
    loss_redundant = logdet_barrier_loss(z_redundant, eps=1e-6)
    loss_independent = logdet_barrier_loss(z_independent, eps=1e-6)
    assert loss_redundant > loss_independent


def test_logdet_barrier_decreases_with_gradient_descent_on_collapsed_channel():
    torch.manual_seed(7)
    z = torch.randn(300, 8, requires_grad=True)
    with torch.no_grad():
        z[:, 0] *= 1e-3
    opt = torch.optim.Adam([z], lr=0.05)
    loss0 = logdet_barrier_loss(z).item()
    for _ in range(50):
        opt.zero_grad()
        loss = logdet_barrier_loss(z)
        loss.backward()
        opt.step()
    loss1 = logdet_barrier_loss(z).item()
    assert loss1 < loss0


def test_logdet_barrier_handles_small_batch_without_nan():
    torch.manual_seed(8)
    z = torch.randn(4, 6, requires_grad=True)
    loss = logdet_barrier_loss(z)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_logdet_barrier_works_under_bfloat16_autocast():
    """Regression test (added 2026-09-01, found via a real crash under
    --amp): torch.linalg.slogdet raises outright on a bfloat16 input
    ('Low precision dtypes not supported') rather than autocast silently
    upcasting it the way softmax/layer_norm are -- logdet_barrier_loss
    must explicitly cast to float32 itself."""
    torch.manual_seed(9)
    z = torch.randn(8, 6, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        z_bf16 = z.to(torch.bfloat16)
        loss = logdet_barrier_loss(z_bf16)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()
