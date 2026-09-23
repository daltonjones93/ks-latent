"""Tests for `temporal_expansion_floor_loss`/`reference_temporal_separation`
(added 2026-09-09, see `ks_latent.training.losses.temporal_expansion_floor_loss`'s
own docstring for the full motivation): a one-sided floor on how close
together, in representation space, real states some fixed number of real
steps apart are allowed to become -- calibrated from the true system's own
real-data separation at that lag, computed independent of the encoder.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.training.losses import reference_temporal_separation, temporal_expansion_floor_loss


def test_reference_temporal_separation_matches_manual_computation():
    torch.manual_seed(0)
    states = torch.randn(3, 10, 5)
    ref = reference_temporal_separation(states, lag=1)
    manual = (states[:, 1:] - states[:, :-1]).norm(dim=-1).mean()
    assert torch.allclose(ref, manual)


def test_reference_temporal_separation_rejects_lag_too_large():
    states = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="must be > lag"):
        reference_temporal_separation(states, lag=3)


def test_floor_loss_penalizes_collapsed_states():
    torch.manual_seed(0)
    real = torch.randn(4, 20, 8).cumsum(dim=1) * 0.5
    ref = reference_temporal_separation(real, lag=1)
    collapsed = real[:, :1].expand(-1, 20, -1)  # exactly constant over time
    loss = temporal_expansion_floor_loss(collapsed, lag=1, sep_ref=ref)
    assert loss.item() > 0.5  # collapsed states have zero separation -> full penalty


def test_floor_loss_is_one_sided_never_penalizes_excess_separation():
    torch.manual_seed(0)
    real = torch.randn(4, 20, 8).cumsum(dim=1) * 0.5
    ref = reference_temporal_separation(real, lag=1)
    exaggerated = real * 5.0  # MORE separated than the reference
    loss = temporal_expansion_floor_loss(exaggerated, lag=1, sep_ref=ref)
    assert loss.item() == 0.0


def test_floor_loss_near_zero_when_matching_reference_distribution():
    torch.manual_seed(0)
    real = torch.randn(4, 20, 8).cumsum(dim=1) * 0.5
    ref = reference_temporal_separation(real, lag=1)
    # a fresh draw from the SAME generative process (not the identical
    # tensor) should incur only a small residual loss, not a large one.
    torch.manual_seed(1)
    real2 = torch.randn(4, 20, 8).cumsum(dim=1) * 0.5
    loss = temporal_expansion_floor_loss(real2, lag=1, sep_ref=ref)
    collapsed_loss = temporal_expansion_floor_loss(real2[:, :1].expand(-1, 20, -1), lag=1, sep_ref=ref)
    assert loss.item() < collapsed_loss.item()


def test_floor_loss_supports_lag_greater_than_1():
    torch.manual_seed(0)
    real = torch.randn(4, 20, 8).cumsum(dim=1) * 0.5
    ref = reference_temporal_separation(real, lag=3)
    loss = temporal_expansion_floor_loss(real, lag=3, sep_ref=ref)
    assert torch.isfinite(loss).all()
    assert loss.item() >= 0.0


def test_floor_loss_rejects_lag_too_large():
    states = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="must be > lag"):
        temporal_expansion_floor_loss(states, lag=5, sep_ref=torch.tensor(1.0))


def test_gradient_flows_through_floor_loss():
    torch.manual_seed(0)
    states = torch.randn(2, 5, 4, requires_grad=True)
    ref = torch.tensor(2.0)  # deliberately large, so the floor is active
    loss = temporal_expansion_floor_loss(states, lag=1, sep_ref=ref)
    loss.backward()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()
