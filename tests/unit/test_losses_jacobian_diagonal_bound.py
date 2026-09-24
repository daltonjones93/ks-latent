"""Tests for `ks_latent.training.losses.propagator_jacobian_diagonal_bound_loss`
(added 2026-09-24, Section 216, user-directed: "it looks like stage 2 is
having trouble converging. If we combined the D3 regularizer with a term
that bounded the magnitude of the diagonal of the jacobian, maybe that
would help"). See that function's docstring for the full motivation
(Section 215's finding that bandedness alone made standalone divergence
worse) and why it's meant to be combined with, not substituted for,
`propagator_jacobian_bandedness_loss`.
"""

from __future__ import annotations

import torch

from ks_latent.training.losses import (
    propagator_jacobian_bandedness_loss,
    propagator_jacobian_diagonal_bound_loss,
)


def _linear_step_fn(W: torch.Tensor):
    def step_fn(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T
    return step_fn


def test_zero_penalty_when_diagonal_within_ceiling():
    d = 10
    W = torch.eye(d) * 1.0  # diagonal exactly at a safe value
    z = torch.randn(16, d)
    loss = propagator_jacobian_diagonal_bound_loss(_linear_step_fn(W), z, ceiling=1.5)
    assert loss.item() == 0.0


def test_penalizes_large_diagonal():
    d = 10
    W_safe = torch.eye(d) * 1.0
    W_explosive = torch.eye(d) * 5.0
    z = torch.randn(16, d)
    loss_safe = propagator_jacobian_diagonal_bound_loss(_linear_step_fn(W_safe), z, ceiling=1.5)
    loss_explosive = propagator_jacobian_diagonal_bound_loss(_linear_step_fn(W_explosive), z, ceiling=1.5)
    assert loss_explosive > loss_safe
    assert loss_safe.item() == 0.0


def test_never_penalizes_contractive_diagonal():
    """One-sided: a diagonal entry BELOW the ceiling (including near-zero
    or negative-magnitude-equivalent, i.e. strongly contractive) must
    never be penalized -- this term should never push toward MORE
    expansion, only cap runaway growth."""
    d = 8
    W_contractive = torch.eye(d) * 0.1
    z = torch.randn(16, d)
    loss = propagator_jacobian_diagonal_bound_loss(_linear_step_fn(W_contractive), z, ceiling=1.5)
    assert loss.item() == 0.0


def test_orthogonal_to_off_diagonal_structure():
    """The whole point of pairing this with bandedness: this loss must
    depend ONLY on the diagonal, not on how coupling is distributed
    off-diagonal -- two matrices with an IDENTICAL (explosive) diagonal
    but very different off-diagonal patterns (one banded/local, one
    dense) must score identically here."""
    torch.manual_seed(0)
    d = 12
    diag_vals = torch.full((d,), 4.0)

    idx = torch.arange(d)
    dist = torch.minimum((idx[:, None] - idx[None, :]).abs(), d - (idx[:, None] - idx[None, :]).abs())
    band_mask = (dist <= 2).float()
    W_banded = torch.randn(d, d) * 0.2 * band_mask
    W_banded.fill_diagonal_(0.0)
    W_banded += torch.diag(diag_vals)

    W_dense = torch.randn(d, d) * 0.2
    W_dense.fill_diagonal_(0.0)
    W_dense += torch.diag(diag_vals)

    z = torch.randn(16, d)
    loss_banded = propagator_jacobian_diagonal_bound_loss(_linear_step_fn(W_banded), z, ceiling=1.5)
    loss_dense = propagator_jacobian_diagonal_bound_loss(_linear_step_fn(W_dense), z, ceiling=1.5)
    assert torch.allclose(loss_banded, loss_dense, atol=1e-5)

    # Sanity: the two DO differ under the bandedness loss, confirming
    # this test's off-diagonal patterns are actually meaningfully
    # different, not accidentally identical.
    band_score = propagator_jacobian_bandedness_loss(_linear_step_fn(W_banded), z)
    dense_score = propagator_jacobian_bandedness_loss(_linear_step_fn(W_dense), z)
    assert band_score < dense_score


def test_loss_decreases_with_gradient_descent_on_an_explosive_diagonal():
    torch.manual_seed(1)
    d = 10
    linear = torch.nn.Linear(d, d, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(d) * 4.0 + torch.randn(d, d) * 0.05)
    opt = torch.optim.Adam(linear.parameters(), lr=0.1)
    z = torch.randn(32, d)

    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = propagator_jacobian_diagonal_bound_loss(linear, z, ceiling=1.5)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
    assert losses[-1] < 5e-2  # should have driven the diagonal most of the way back under the ceiling


def test_output_is_scalar_and_finite():
    d = 10
    z = torch.randn(16, d)
    loss = propagator_jacobian_diagonal_bound_loss(_linear_step_fn(torch.randn(d, d) * 0.2), z)
    assert loss.shape == ()
    assert torch.isfinite(loss)
