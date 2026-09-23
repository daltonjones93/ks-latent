"""Tests for the LR warmup/floor schedule and the L1/L2 latent-loss switch,
added 2026-08-29 (ported from a reference implementation, see
CLAUDE_CODE_BRIEF.md §5.1 "Ported improvements" addendum)."""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import Stage2TrainingConfig
from ks_latent.training.loops import k_curriculum, warmup_cosine_lr_lambda
from ks_latent.training.losses import horizon_weighted_latent_loss, horizon_weights


def test_k_curriculum_ramps_linearly_from_k_min_to_k_max():
    values = [k_curriculum(e, k_min=2, k_max=8, warmup_epochs=6) for e in range(7)]
    assert values == [2, 3, 4, 5, 6, 7, 8]


def test_k_curriculum_holds_at_k_max_after_warmup():
    assert k_curriculum(100, k_min=2, k_max=8, warmup_epochs=6) == 8
    assert k_curriculum(6, k_min=2, k_max=8, warmup_epochs=6) == 8


def test_k_curriculum_zero_warmup_holds_at_k_max_from_epoch_zero():
    assert k_curriculum(0, k_min=2, k_max=8, warmup_epochs=0) == 8
    assert k_curriculum(5, k_min=2, k_max=8, warmup_epochs=0) == 8


def test_k_curriculum_stage2_default_starts_at_two():
    """Stage 2's existing curriculum is k_curriculum with k_min=2 -- pin
    that equivalence so the two never silently diverge."""
    from ks_latent.training.loops import _k_curriculum

    for epoch in range(10):
        assert _k_curriculum(epoch, k_max=8, warmup_epochs=6) == k_curriculum(
            epoch, k_min=2, k_max=8, warmup_epochs=6
        )


def test_warmup_ramps_linearly_to_one():
    f = warmup_cosine_lr_lambda(epochs=10, warmup_epochs=4, lr_min_factor=0.02)
    factors = [f(e) for e in range(4)]
    assert factors == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_cosine_decay_after_warmup_approaches_floor_monotonically():
    f = warmup_cosine_lr_lambda(epochs=10, warmup_epochs=2, lr_min_factor=0.02)
    assert f(1) == pytest.approx(1.0)  # last warmup epoch hits full LR
    assert f(9) < 0.1  # close to the floor by the last epoch
    assert f(9) < f(5) < f(1)  # monotonically decaying in between


def test_epochs_plus_one_reaches_the_floor_exactly():
    """`prog` reaches exactly 1.0 (the floor) one epoch past `epochs - 1`;
    schedules are always sized with enough epochs that training stops before
    this, but the limit itself should be exact."""
    f = warmup_cosine_lr_lambda(epochs=10, warmup_epochs=2, lr_min_factor=0.02)
    assert f(10) == pytest.approx(0.02, abs=1e-9)


def test_zero_warmup_epochs_is_pure_cosine_from_epoch_zero():
    f = warmup_cosine_lr_lambda(epochs=10, warmup_epochs=0, lr_min_factor=0.0)
    assert f(0) == pytest.approx(1.0)
    assert f(9) < 0.05


def test_schedule_never_exceeds_one_or_drops_below_floor():
    f = warmup_cosine_lr_lambda(epochs=20, warmup_epochs=3, lr_min_factor=0.05)
    values = [f(e) for e in range(20)]
    assert max(values) <= 1.0 + 1e-9
    assert min(values) >= 0.05 - 1e-9


def test_stage2_config_rejects_invalid_latent_loss():
    with pytest.raises(ValueError):
        Stage2TrainingConfig(latent_loss="huber")


def test_horizon_weighted_latent_loss_l2_matches_mse():
    torch.manual_seed(0)
    z_pred = torch.randn(8, 3, 5)
    z_true = torch.randn(8, 3, 5)
    w = horizon_weights(3, gamma=1.0)
    l2 = horizon_weighted_latent_loss(z_pred, z_true, w, kind="l2")
    manual = (w * ((z_pred - z_true) ** 2).mean(dim=(0, 2))).sum()
    assert l2.item() == pytest.approx(manual.item(), rel=1e-5)


def test_horizon_weighted_latent_loss_l1_less_sensitive_to_outlier():
    """A single huge-error sample inflates L2 far more than L1 -- exactly the
    'less dominated by rare large increments' property motivating the L1
    option (Stage2TrainingConfig.latent_loss docstring)."""
    z_true = torch.zeros(8, 1, 4)
    z_pred = torch.zeros(8, 1, 4)
    z_pred[0] = 100.0  # one wild outlier sample
    w = horizon_weights(1, gamma=1.0)
    l1 = horizon_weighted_latent_loss(z_pred, z_true, w, kind="l1").item()
    l2 = horizon_weighted_latent_loss(z_pred, z_true, w, kind="l2").item()
    # Same data, so compare each to what the *other* metric would give a
    # uniformly-scaled-down error of the same L1 norm: l2/l1 ratio should be
    # much larger than 1, showing L2's outlier sensitivity.
    assert l2 / l1 > 10.0


def test_horizon_weighted_latent_loss_rejects_unknown_kind():
    z = torch.zeros(2, 1, 3)
    w = horizon_weights(1)
    with pytest.raises(ValueError):
        horizon_weighted_latent_loss(z, z, w, kind="huber")
