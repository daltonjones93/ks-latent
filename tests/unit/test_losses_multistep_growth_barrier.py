"""Tests for `ks_latent.training.losses.propagator_multistep_growth_barrier_loss`
(added 2026-09-24, Section 220, user-directed after Section 219's squared-
hinge ceiling still let `masked_mlp` diverge FASTER than Section 218's
unconstrained baseline: "lower the 150 bound and penalize this
differently. instead of using mse, use some kind of -log loss such that
there is a boundary at wherever we want to bound the jacobian"). See that
function's docstring for the full safeguarded-log-barrier construction.
"""

from __future__ import annotations

import math

import torch

from ks_latent.training.losses import (
    propagator_multistep_growth_barrier_loss,
    propagator_multistep_growth_ceiling_loss,
)


def _linear_step_fn(W: torch.Tensor):
    def step_fn(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T
    return step_fn


def test_small_loss_well_below_ceiling():
    d = 10
    W = torch.eye(d) * 1.0  # k-step growth stays 1.0, far below ceiling
    z = torch.randn(16, d)
    loss = propagator_multistep_growth_barrier_loss(_linear_step_fn(W), z, k=10, ceiling=75.0)
    # -log(75-1) ~ -log(74) ~ -4.3 -- a real finite, small-magnitude value,
    # not exploding, and clearly less than the value near the boundary.
    assert torch.isfinite(loss)
    assert loss.item() < -3.0


def test_loss_grows_sharply_as_growth_approaches_ceiling():
    """The whole point of a barrier over a hinge: the penalty must
    increase (not stay at zero) as growth approaches the ceiling from
    below, well before it is ever crossed."""
    d = 6
    ceiling = 75.0
    z = torch.randn(16, d)
    # Scales whose k=10 growth stays under ceiling but gets closer to it.
    scales = [1.0, 1.30, 1.38, 1.40]  # 10-step growth ~= [1, 13.8, 26.1, 28.9]
    losses = []
    for s in scales:
        W = torch.eye(d) * s
        loss = propagator_multistep_growth_barrier_loss(_linear_step_fn(W), z, k=10, ceiling=ceiling)
        losses.append(loss.item())
    assert losses == sorted(losses)  # strictly increasing as growth approaches ceiling
    assert losses[-1] > losses[0]


def test_never_nan_or_inf_even_far_past_ceiling():
    d = 8
    z = torch.randn(16, d)
    ceiling = 10.0
    # k=10 growth for scale=3.0 is 3**10 ~ 59049, WAY past ceiling=10.
    W_explosive = torch.eye(d) * 3.0
    loss = propagator_multistep_growth_barrier_loss(_linear_step_fn(W_explosive), z, k=10, ceiling=ceiling)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0  # past the switch point, should read as a real, large-ish penalty


def test_c1_continuity_at_the_safeguard_switch_point():
    """Value AND derivative (w.r.t. the scalar diagonal scale, evaluated
    via finite differences) should be continuous across the epsilon
    switch -- verifies the linear extrapolation matches the log-barrier's
    own tangent line at margin=epsilon, not just its value."""
    d = 4
    ceiling = 20.0
    epsilon = 1.0  # switch point at margin=1.0, i.e. growth=19.0
    z = torch.randn(8, d)

    def loss_at_growth(growth: float) -> float:
        # A diagonal matrix whose k=1 step already equals `growth` directly
        # (k=1 avoids needing to invert growth**(1/k) for the test).
        W = torch.eye(d) * growth
        return propagator_multistep_growth_barrier_loss(
            _linear_step_fn(W), z, k=1, ceiling=ceiling, epsilon=epsilon
        ).item()

    switch_growth = ceiling - epsilon  # = 19.0
    h = 1e-4
    val_below = loss_at_growth(switch_growth - h)
    val_at = loss_at_growth(switch_growth)
    val_above = loss_at_growth(switch_growth + h)

    # Value continuity: both sides should closely bracket the exact value.
    assert abs(val_below - val_at) < 1e-2
    assert abs(val_above - val_at) < 1e-2

    # Derivative continuity: finite-difference slope on each side should
    # both be close to the analytic tangent slope -1/epsilon (d loss / d
    # growth = d loss / d margin * d margin / d growth = (-1/epsilon)*(-1)
    # = 1/epsilon here, since margin = ceiling - growth).
    slope_below = (val_at - val_below) / h
    slope_above = (val_above - val_at) / h
    expected_slope = 1.0 / epsilon
    assert abs(slope_below - expected_slope) < 0.05
    assert abs(slope_above - expected_slope) < 0.05


def test_gradient_is_nonzero_below_ceiling_unlike_the_squared_hinge():
    """The squared-hinge ceiling loss has EXACTLY ZERO gradient anywhere
    below `ceiling` -- confirm the barrier version does NOT share that
    blind spot: even comfortably under the ceiling, gradient must be
    nonzero (repelling approach), while the hinge's gradient there is
    exactly zero."""
    d = 6
    ceiling = 75.0
    z = torch.randn(16, d)
    linear = torch.nn.Linear(d, d, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(d) * 1.3)  # 10-step growth ~13.8, well under 75

    barrier_loss = propagator_multistep_growth_barrier_loss(linear, z, k=10, ceiling=ceiling)
    barrier_loss.backward()
    barrier_grad_norm = linear.weight.grad.norm().item()
    linear.weight.grad = None

    hinge_loss = propagator_multistep_growth_ceiling_loss(linear, z, k=10, ceiling=ceiling)
    hinge_loss.backward()
    hinge_grad_norm = linear.weight.grad.norm().item()

    assert hinge_loss.item() == 0.0
    assert hinge_grad_norm == 0.0
    assert barrier_grad_norm > 0.0


def test_loss_decreases_with_gradient_descent_on_an_explosive_map():
    torch.manual_seed(1)
    d = 8
    linear = torch.nn.Linear(d, d, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(d) * 2.0 + torch.randn(d, d) * 0.02)
    opt = torch.optim.Adam(linear.parameters(), lr=0.02)
    z = torch.randn(16, d)

    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = propagator_multistep_growth_barrier_loss(linear, z, k=10, ceiling=75.0)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < losses[0]


def test_output_is_scalar_and_finite():
    d = 10
    z = torch.randn(16, d)
    loss = propagator_multistep_growth_barrier_loss(
        _linear_step_fn(torch.randn(d, d) * 0.2), z, k=10, ceiling=75.0
    )
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_default_epsilon_is_five_percent_of_ceiling():
    """epsilon=None should behave identically to explicitly passing
    0.05*ceiling."""
    d = 6
    ceiling = 40.0
    z = torch.randn(12, d)
    W = torch.eye(d) * 1.35
    loss_default = propagator_multistep_growth_barrier_loss(_linear_step_fn(W), z, k=10, ceiling=ceiling)
    loss_explicit = propagator_multistep_growth_barrier_loss(
        _linear_step_fn(W), z, k=10, ceiling=ceiling, epsilon=0.05 * ceiling
    )
    assert torch.allclose(loss_default, loss_explicit)
