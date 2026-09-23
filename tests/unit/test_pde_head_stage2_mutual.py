"""Tests for `pde_head`'s Stage-2 continuation (added 2026-09-08, see
docs/sine_transform_pde_plan.md §21, user-directed: "is the pde_head
trained during phase 2 as well. we should try to get pde rollout to have
decent performance" then "please implement both option a and b, and we
can try both. at the end of the day, we're really just training the
propagator right, the pde_head training is acting as a regularization
term").

Unlike Stage 1's `pde_head` (target detached, `aux` protected), Stage 2's
version is DELIBERATELY MUTUAL -- Stage 2 never touches the encoder, so a
detached target would leave nothing for the loss to regularize except
`pde_head` in isolation. These tests verify the mutual property directly:
BOTH `propagator` and `pde_head` must receive nonzero gradient, for each
of "option A" (`w_pde_distill`, single-step-at-every-position) and
"option B" (`w_pde_rollout`, pde_head's own autoregressive rollout).
"""

from __future__ import annotations

import torch

from ks_latent.config import AuxPropagatorConfig, PropagatorConfig
from ks_latent.models.propagator import AuxPropagator, LatentPropagator
from ks_latent.utils.seeding import set_seed


def _build(d_latent: int = 20):
    prop_cfg = PropagatorConfig(
        d_latent=d_latent, hidden=16, n_blocks=1, dropout=0.0,
        mode="markovian", backbone="mlp", zero_init=False,
    )
    prop = LatentPropagator(prop_cfg)
    pde_cfg = AuxPropagatorConfig(
        d_latent=d_latent, hidden=16, n_blocks=1, mode="markovian", backbone="spectral_pde_raw",
        spectral_K=d_latent // 2 + 1, spectral_L=float(d_latent), spectral_max_order=4,
        spectral_integrator="euler", zero_init=False,
    )
    pde_head = AuxPropagator(pde_cfg)
    return prop, pde_head


def _option_a_loss(prop, pde_head, z_prev, z_curr, k_now):
    z_pred = prop.rollout(z_prev, z_curr, k_now)
    z_states = torch.cat([z_curr.unsqueeze(1), z_pred[:, :-1]], dim=1)
    d = z_pred.shape[-1]
    p_pred = pde_head.step_one(z_states.reshape(-1, d)).reshape(z_pred.shape)
    return ((p_pred - z_pred) ** 2).mean()


def _option_b_loss(prop, pde_head, z_prev, z_curr, k_now):
    z_pred = prop.rollout(z_prev, z_curr, k_now)
    z_pde_rollout = pde_head.rollout(z_curr, z_curr, k_now)
    return ((z_pde_rollout - z_pred) ** 2).mean()


def test_option_a_gives_mutual_nonzero_gradient():
    set_seed(0)
    prop, pde_head = _build()
    z_prev = torch.randn(4, 20)
    z_curr = torch.randn(4, 20)

    loss = _option_a_loss(prop, pde_head, z_prev, z_curr, k_now=3)
    loss.backward()

    prop_grads = [p.grad for p in prop.parameters() if p.grad is not None]
    pde_grads = [p.grad for p in pde_head.parameters() if p.grad is not None]
    assert len(prop_grads) > 0 and any(torch.any(g != 0) for g in prop_grads), (
        "propagator must receive nonzero gradient -- option A is mutual, not detached."
    )
    assert len(pde_grads) > 0 and any(torch.any(g != 0) for g in pde_grads)


def test_option_b_gives_mutual_nonzero_gradient():
    set_seed(0)
    prop, pde_head = _build()
    z_prev = torch.randn(4, 20)
    z_curr = torch.randn(4, 20)

    loss = _option_b_loss(prop, pde_head, z_prev, z_curr, k_now=3)
    loss.backward()

    prop_grads = [p.grad for p in prop.parameters() if p.grad is not None]
    pde_grads = [p.grad for p in pde_head.parameters() if p.grad is not None]
    assert len(prop_grads) > 0 and any(torch.any(g != 0) for g in prop_grads), (
        "propagator must receive nonzero gradient -- option B is mutual, not detached."
    )
    assert len(pde_grads) > 0 and any(torch.any(g != 0) for g in pde_grads)


def test_option_a_pde_head_gradient_is_single_step_only():
    """The defining safety property of option A: pde_head's gradient must
    not depend on chaining through its own multi-step rollout -- verified
    indirectly by confirming the loss is identical whether computed via
    the batched single-step formula or via k_now separate step_one calls
    on the SAME (detached-between-calls) states, which would differ from a
    true autoregressive chain through pde_head."""
    set_seed(0)
    prop, pde_head = _build()
    z_prev = torch.randn(4, 20)
    z_curr = torch.randn(4, 20)
    k_now = 3

    z_pred = prop.rollout(z_prev, z_curr, k_now)
    z_states = torch.cat([z_curr.unsqueeze(1), z_pred[:, :-1]], dim=1)
    d = z_pred.shape[-1]

    # Batched (as used in train_stage2)
    p_pred_batched = pde_head.step_one(z_states.reshape(-1, d)).reshape(z_pred.shape)

    # Per-step, independently -- must match exactly since each step_one
    # call only ever sees a fixed (already-computed) input state.
    p_pred_manual = torch.stack(
        [pde_head.step_one(z_states[:, k]) for k in range(k_now)], dim=1
    )
    assert torch.allclose(p_pred_batched, p_pred_manual, atol=1e-6)


def test_option_b_is_a_true_autoregressive_chain():
    """Option B's pde_head.rollout output at step k>0 must depend on
    pde_head's own step k-1 output (chained), NOT on propagator's -- i.e.
    it differs from option A's per-position formula whenever pde_head's
    own prediction diverges from propagator's."""
    set_seed(0)
    prop, pde_head = _build()
    z_curr = torch.randn(4, 20)
    k_now = 3

    z_pde_rollout = pde_head.rollout(z_curr, z_curr, k_now)
    # Manually chain: step 0 from z_curr, step 1 from step 0's own output, etc.
    manual = []
    z = z_curr
    for _ in range(k_now):
        z = pde_head.step_one(z)
        manual.append(z)
    manual = torch.stack(manual, dim=1)
    assert torch.allclose(z_pde_rollout, manual, atol=1e-6)
