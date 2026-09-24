"""Tests for `ks_latent.training.losses.propagator_rollout_magnitude_ceiling_loss`
(added 2026-09-24, Section 218, user-directed: "I think we need to bound
the growth of the propagator during stage 1"). See that function's
docstring for the full motivation (Section 217's masked_mlp propagator
diverging to max|z|~1e30 with nothing in Stage 1's objective penalizing
it) and its relationship to `spatial_energy_floor_loss` (a floor on a
different quantity -- this is the missing ceiling on raw magnitude).
"""

from __future__ import annotations

import torch

from ks_latent.training.losses import propagator_rollout_magnitude_ceiling_loss


def test_zero_penalty_when_within_ceiling():
    z = torch.randn(8, 5, 10) * 0.5  # small values, well under any reasonable ceiling
    loss = propagator_rollout_magnitude_ceiling_loss(z.reshape(-1, 10), ceiling=15.0)
    assert loss.item() == 0.0


def test_penalizes_large_magnitude():
    z_small = torch.full((4, 10), 5.0)
    z_large = torch.full((4, 10), 100.0)
    loss_small = propagator_rollout_magnitude_ceiling_loss(z_small, ceiling=15.0)
    loss_large = propagator_rollout_magnitude_ceiling_loss(z_large, ceiling=15.0)
    assert loss_small.item() == 0.0
    assert loss_large > loss_small


def test_never_penalizes_small_magnitude():
    """One-sided: values well under the ceiling (including near zero)
    must never be penalized -- this term should never push toward
    LARGER magnitude, only cap runaway growth."""
    z = torch.zeros(4, 10)
    loss = propagator_rollout_magnitude_ceiling_loss(z, ceiling=15.0)
    assert loss.item() == 0.0


def test_uses_max_abs_across_last_dim_not_norm():
    """Confirms the statistic is max|z| (matching this project's own
    standalone-rollout diagnostic convention), not e.g. an L2 norm --
    a vector with one large entry and many zeros should be penalized
    the same as a vector with that same large entry repeated, since
    both have the same max|.|."""
    d = 20
    z_one_spike = torch.zeros(1, d)
    z_one_spike[0, 0] = 50.0
    z_all_spike = torch.full((1, d), 50.0)
    loss_one = propagator_rollout_magnitude_ceiling_loss(z_one_spike, ceiling=15.0)
    loss_all = propagator_rollout_magnitude_ceiling_loss(z_all_spike, ceiling=15.0)
    assert torch.allclose(loss_one, loss_all)


def test_loss_decreases_with_gradient_descent():
    """A single-parameter (scalar magnitude) optimization: cleaner than
    a multi-dim tensor, where `amax`'s sub-gradient only ever touches
    the current argmax entry and convergence rate depends heavily on
    optimizer/LR specifics unrelated to the loss's own correctness."""
    torch.manual_seed(0)
    scale = torch.tensor(30.0, requires_grad=True)
    opt = torch.optim.Adam([scale], lr=1.0)
    losses = []
    for _ in range(100):
        opt.zero_grad()
        z = torch.full((4, 10), 1.0) * scale
        loss = propagator_rollout_magnitude_ceiling_loss(z, ceiling=15.0)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert losses[-1] < 1.0


def test_output_is_scalar_and_finite():
    z = torch.randn(4, 3, 10) * 5.0
    loss = propagator_rollout_magnitude_ceiling_loss(z.reshape(-1, 10), ceiling=15.0)
    assert loss.shape == ()
    assert torch.isfinite(loss)
