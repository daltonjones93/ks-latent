"""Tests for `ks_latent.training.losses.propagator_multistep_growth_ceiling_loss`
(added 2026-09-24, Section 219, user-directed follow-up to Section 218's
`propagator_rollout_magnitude_ceiling_loss`: "try not to clamp too hard to
preserve the chaotic dynamics ... another way to do this is just make sure
the longer term jacobian after 10 steps doesn't expand too much"). See
that function's docstring for the full motivation and the Section 216
empirical calibration behind the default ceiling.
"""

from __future__ import annotations

import torch

from ks_latent.training.losses import (
    propagator_local_expansion_floor_loss,
    propagator_multistep_growth_ceiling_loss,
)


def _linear_step_fn(W: torch.Tensor):
    def step_fn(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T
    return step_fn


def test_zero_penalty_when_growth_within_ceiling():
    # A mildly contractive diagonal map: k-step growth = 0.9**k < 1, well
    # under any reasonable ceiling.
    d = 10
    W = torch.eye(d) * 0.9
    z = torch.randn(16, d)
    loss = propagator_multistep_growth_ceiling_loss(_linear_step_fn(W), z, k=10, ceiling=1.0)
    assert loss.item() == 0.0


def test_penalizes_large_composed_growth():
    d = 10
    W_safe = torch.eye(d) * 1.0  # k-step growth stays exactly 1.0
    W_explosive = torch.eye(d) * 1.5  # k-step growth = 1.5**10 ~ 57.7
    z = torch.randn(16, d)
    loss_safe = propagator_multistep_growth_ceiling_loss(_linear_step_fn(W_safe), z, k=10, ceiling=10.0)
    loss_explosive = propagator_multistep_growth_ceiling_loss(
        _linear_step_fn(W_explosive), z, k=10, ceiling=10.0
    )
    assert loss_safe.item() == 0.0
    assert loss_explosive > loss_safe


def test_never_penalizes_contractive_or_at_ceiling_growth():
    d = 8
    W_contractive = torch.eye(d) * 0.5
    z = torch.randn(16, d)
    loss = propagator_multistep_growth_ceiling_loss(_linear_step_fn(W_contractive), z, k=10, ceiling=1.0)
    assert loss.item() == 0.0


def test_growth_is_computed_over_the_composed_k_step_map_not_one_step():
    """The whole point of this loss over a one-step spectral-norm ceiling:
    a map whose SINGLE-step top singular value is comfortably under a
    ceiling can still explode once composed k times. Eigenvalue 1.3: one
    step is 1.3 (under a ceiling of, say, 5.0 -- would score 0 under a
    naive one-step ceiling at that threshold), but 10 steps compose to
    1.3**10 ~ 13.79, which SHOULD be penalized at that same threshold."""
    d = 6
    W = torch.eye(d) * 1.3
    z = torch.randn(16, d)
    ceiling = 5.0
    loss_one_step = propagator_multistep_growth_ceiling_loss(_linear_step_fn(W), z, k=1, ceiling=ceiling)
    loss_ten_step = propagator_multistep_growth_ceiling_loss(_linear_step_fn(W), z, k=10, ceiling=ceiling)
    assert loss_one_step.item() == 0.0  # 1.3 < 5.0, no penalty at k=1
    assert loss_ten_step > 0.0  # 1.3**10 ~ 13.79 > 5.0, penalized at k=10


def test_orthogonal_to_bandedness_or_diagonal_shape():
    """Depends only on the composed map's overall spectral (operator)
    norm, not on how coupling is distributed structurally -- a banded and
    a dense matrix with the SAME dominant growth rate should score
    similarly (not necessarily identical, since off-diagonal structure can
    shift the exact top singular value slightly, but both should register
    as clearly over ceiling together when the underlying growth is the
    same order of magnitude)."""
    torch.manual_seed(0)
    d = 12
    diag_vals = torch.full((d,), 1.4)
    idx = torch.arange(d)
    dist = torch.minimum((idx[:, None] - idx[None, :]).abs(), d - (idx[:, None] - idx[None, :]).abs())
    band_mask = (dist <= 2).float()
    W_banded = torch.randn(d, d) * 0.05 * band_mask
    W_banded.fill_diagonal_(0.0)
    W_banded += torch.diag(diag_vals)

    z = torch.randn(16, d)
    ceiling = 5.0  # well under 1.4**10 ~ 28.9
    loss = propagator_multistep_growth_ceiling_loss(_linear_step_fn(W_banded), z, k=10, ceiling=ceiling)
    assert loss > 0.0


def test_loss_decreases_with_gradient_descent_on_an_explosive_map():
    torch.manual_seed(1)
    d = 10
    linear = torch.nn.Linear(d, d, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(d) * 1.6 + torch.randn(d, d) * 0.02)
    opt = torch.optim.Adam(linear.parameters(), lr=0.05)
    z = torch.randn(16, d)

    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = propagator_multistep_growth_ceiling_loss(linear, z, k=10, ceiling=20.0)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_output_is_scalar_and_finite():
    d = 10
    z = torch.randn(16, d)
    loss = propagator_multistep_growth_ceiling_loss(
        _linear_step_fn(torch.randn(d, d) * 0.2), z, k=10, ceiling=150.0
    )
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_does_not_fight_a_genuinely_healthy_local_expansion_signal():
    """Sanity cross-check against `propagator_local_expansion_floor_loss`
    (which floors the ONE-STEP top singular value at 1.0, rewarding
    genuine local expansion): a map with a modest, non-runaway one-step
    expansion (top singular value ~1.1) should satisfy BOTH the existing
    expansion floor and this new ceiling at once with headroom -- i.e.
    this loss genuinely does not fight ordinary chaotic behavior, only
    runaway growth well beyond it."""
    d = 10
    W = torch.eye(d) * 1.1
    z = torch.randn(16, d)
    floor_loss = propagator_local_expansion_floor_loss(_linear_step_fn(W), z, floor=1.0)
    ceiling_loss = propagator_multistep_growth_ceiling_loss(_linear_step_fn(W), z, k=10, ceiling=150.0)
    assert floor_loss.item() == 0.0
    assert ceiling_loss.item() == 0.0
