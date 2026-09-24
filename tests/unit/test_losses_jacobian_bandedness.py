"""Tests for `ks_latent.training.losses.propagator_jacobian_bandedness_loss`
(added 2026-09-24, user-directed: "let's make D3 into a loss, since this
seems like it's the most important statistic to improve our chances at
being able to localize as in 4.3 in the literature review document"). See
that function's docstring for the full motivation and the precedent it
mirrors (`spatial_coherence_loss`, D7's own differentiable version).
"""

from __future__ import annotations

import torch

from ks_latent.training.losses import propagator_jacobian_bandedness_loss


def _circular_band_weight(d: int, radius: int, off_value: float = 0.0) -> torch.Tensor:
    """`(d, d)` weight matrix, nonzero only within circular distance
    `radius` (off-diagonal entries set to a fixed nonzero constant so the
    Jacobian has real, measurable off-diagonal mass -- an all-zero
    off-band Jacobian would trivially fall into the eps-regularized
    fallback rather than actually testing bandedness)."""
    idx = torch.arange(d)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    dist = torch.minimum(diff, d - diff)
    W = torch.where(dist <= radius, torch.full((d, d), 0.5), torch.full((d, d), off_value))
    W.fill_diagonal_(1.0)
    return W


def _linear_step_fn(W: torch.Tensor):
    def step_fn(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T
    return step_fn


def test_loss_is_lower_for_banded_jacobian_than_dense():
    torch.manual_seed(0)
    d = 16
    W_banded = _circular_band_weight(d, radius=2)
    W_dense = torch.randn(d, d) * 0.3
    W_dense.fill_diagonal_(1.0)

    z = torch.randn(32, d)
    loss_banded = propagator_jacobian_bandedness_loss(_linear_step_fn(W_banded), z)
    loss_dense = propagator_jacobian_bandedness_loss(_linear_step_fn(W_dense), z)
    assert loss_banded < loss_dense


def test_wider_band_scores_worse_than_narrower_band():
    """Monotonicity sanity check: a strictly wider coupling radius should
    score strictly worse (higher loss) than a narrower one, holding the
    off-band coupling magnitude fixed."""
    torch.manual_seed(1)
    d = 20
    W_narrow = _circular_band_weight(d, radius=1)
    W_wide = _circular_band_weight(d, radius=6)
    z = torch.randn(32, d)
    loss_narrow = propagator_jacobian_bandedness_loss(_linear_step_fn(W_narrow), z)
    loss_wide = propagator_jacobian_bandedness_loss(_linear_step_fn(W_wide), z)
    assert loss_narrow < loss_wide


def test_no_off_diagonal_coupling_is_not_a_spurious_optimum():
    """The identity map (Jacobian = I, zero off-diagonal coupling
    anywhere) must NOT score as a perfect/near-zero loss -- a propagator
    that couples nothing to its neighbors is not "maximally local," it is
    uninformative about locality entirely, and the eps-regularized
    fallback (same mechanism spatial_coherence_loss uses for D7) should
    return the uninformative baseline instead of a spuriously optimal
    score. Compared against a genuinely banded-but-coupled propagator,
    which should score BETTER (lower loss) than the fallback baseline."""
    d = 16
    z = torch.randn(32, d)
    identity_step = _linear_step_fn(torch.eye(d))
    loss_identity = propagator_jacobian_bandedness_loss(identity_step, z)

    W_banded = _circular_band_weight(d, radius=2)
    loss_banded = propagator_jacobian_bandedness_loss(_linear_step_fn(W_banded), z)
    assert loss_banded < loss_identity


def test_loss_decreases_with_gradient_descent_on_a_dense_linear_map():
    """Sanity: the loss is actually differentiable through the
    propagator's own parameters (not just through z) -- a few steps of
    gradient descent on a trainable linear map's weight should reduce
    it, starting from a dense (non-banded) initialization."""
    torch.manual_seed(2)
    d = 12
    linear = torch.nn.Linear(d, d, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(d, d) * 0.3 + torch.eye(d))
    opt = torch.optim.Adam(linear.parameters(), lr=0.05)
    z = torch.randn(32, d)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = propagator_jacobian_bandedness_loss(linear, z)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_output_is_scalar_and_finite():
    d = 10
    z = torch.randn(16, d)
    loss = propagator_jacobian_bandedness_loss(_linear_step_fn(torch.randn(d, d) * 0.2), z)
    assert loss.shape == ()
    assert torch.isfinite(loss)
