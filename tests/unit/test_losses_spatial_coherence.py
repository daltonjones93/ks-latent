"""Tests for `ks_latent.training.losses.spatial_coherence_loss` (added
2026-08-31, user-directed: "turn the D7 diagnostic into a loss ... push the
model towards creating spatial coherence"). See that function's docstring
for the full precedent/risk discussion (`RegConfig.lambda_z`'s known
collapse, and why this mechanism is different but not proven safe).
"""

from __future__ import annotations

import torch

from ks_latent.training.losses import spatial_coherence_loss


def _ring_correlated_batch(d: int, n: int, rho: float, noise_std: float, generator: torch.Generator) -> torch.Tensor:
    """`(n, d)` batch with a circulant covariance (`Cov[i,j] =
    rho^circular_distance(i,j)`) -- genuine local structure, not collapse."""
    idx = torch.arange(d)
    dist = torch.minimum((idx[:, None] - idx[None, :]).abs(), d - (idx[:, None] - idx[None, :]).abs())
    cov = rho ** dist.float()
    L = torch.linalg.cholesky(cov)
    z = torch.randn(n, d, generator=generator) @ L.T
    z += noise_std * torch.randn(n, d, generator=generator)
    return z


def test_loss_is_lower_for_genuinely_banded_data_than_independent_noise():
    g = torch.Generator().manual_seed(0)
    z_banded = _ring_correlated_batch(d=20, n=500, rho=0.85, noise_std=0.05, generator=g)
    z_indep = torch.randn(500, 20, generator=g)
    loss_banded = spatial_coherence_loss(z_banded)
    loss_indep = spatial_coherence_loss(z_indep)
    assert loss_banded < loss_indep


def test_full_collapse_is_not_the_global_optimum():
    """The key safety property distinguishing this from `BandedSmoothness`
    (`RegConfig.lambda_z`), whose unconstrained optimum IS full collapse:
    a fully redundant/collapsed batch (every channel an identical copy of
    one shared signal) must score WORSE (higher loss) than a genuinely
    locally-correlated-but-diverse batch, not better. If this test ever
    fails, `spatial_coherence_loss` has acquired the same degenerate
    global optimum `lambda_z` is already known to collapse the latent via,
    and must not be used as-is."""
    g = torch.Generator().manual_seed(1)
    d, n = 20, 500
    shared_signal = torch.randn(n, 1, generator=g)
    z_collapsed = shared_signal.expand(n, d) + 1e-4 * torch.randn(n, d, generator=g)
    z_banded = _ring_correlated_batch(d=d, n=n, rho=0.85, noise_std=0.05, generator=g)
    loss_collapsed = spatial_coherence_loss(z_collapsed)
    loss_banded = spatial_coherence_loss(z_banded)
    assert loss_collapsed > loss_banded


def test_loss_decreases_with_gradient_descent_on_random_init():
    """Sanity: the loss is actually differentiable and optimizable -- a few
    steps of gradient descent on a random z should reduce it."""
    torch.manual_seed(2)
    z = torch.randn(200, 16, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.05)
    loss0 = spatial_coherence_loss(z).item()
    for _ in range(50):
        opt.zero_grad()
        loss = spatial_coherence_loss(z)
        loss.backward()
        opt.step()
    loss1 = spatial_coherence_loss(z).item()
    assert loss1 < loss0


def test_loss_in_valid_range():
    torch.manual_seed(3)
    z = torch.randn(100, 10)
    loss = spatial_coherence_loss(z)
    assert 0.0 <= loss.item() <= 1.0


def test_handles_small_batch_without_nan():
    torch.manual_seed(4)
    z = torch.randn(4, 8, requires_grad=True)
    loss = spatial_coherence_loss(z)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()


# ---- signed=True (added 2026-09-01, user-directed, backs D8) ----


def test_signed_false_is_invariant_to_per_channel_sign_flips():
    """Default (unsigned) behavior is unchanged: |correlation| doesn't
    care about a per-channel sign flip."""
    g = torch.Generator().manual_seed(5)
    d, n = 20, 500
    z = _ring_correlated_batch(d=d, n=n, rho=0.85, noise_std=0.05, generator=g)
    sign_pattern = torch.tensor([(-1.0) ** i for i in range(d)])
    z_flipped = z * sign_pattern
    loss_orig = spatial_coherence_loss(z, signed=False)
    loss_flipped = spatial_coherence_loss(z_flipped, signed=False)
    assert torch.allclose(loss_orig, loss_flipped, atol=1e-5)


def test_signed_true_distinguishes_same_sign_from_alternating_sign_neighbours():
    """The key property: signed=True must NOT be invariant to the same
    per-channel sign flip -- an alternating-sign copy of a genuinely
    same-sign-coherent batch (every immediate-neighbor pair now
    anti-correlated) must score WORSE (higher loss) than the original,
    even though signed=False scores them identically (previous test)."""
    g = torch.Generator().manual_seed(6)
    d, n = 20, 500
    z = _ring_correlated_batch(d=d, n=n, rho=0.85, noise_std=0.05, generator=g)
    sign_pattern = torch.tensor([(-1.0) ** i for i in range(d)])
    z_flipped = z * sign_pattern
    loss_same_sign = spatial_coherence_loss(z, signed=True)
    loss_alternating = spatial_coherence_loss(z_flipped, signed=True)
    assert loss_same_sign < loss_alternating


def test_signed_true_loss_decreases_with_gradient_descent():
    torch.manual_seed(7)
    z = torch.randn(200, 16, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.05)
    loss0 = spatial_coherence_loss(z, signed=True).item()
    for _ in range(50):
        opt.zero_grad()
        loss = spatial_coherence_loss(z, signed=True)
        loss.backward()
        opt.step()
    loss1 = spatial_coherence_loss(z, signed=True).item()
    assert loss1 < loss0


def test_signed_true_handles_small_batch_without_nan():
    torch.manual_seed(8)
    z = torch.randn(4, 8, requires_grad=True)
    loss = spatial_coherence_loss(z, signed=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()
