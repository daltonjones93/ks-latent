"""Tests for `train_stage2(..., freeze_propagator=True)` ("Phase 3",
added 2026-09-08, see docs/sine_transform_pde_plan.md §23, user-directed:
"should we add a phase 3 that refines the pde with the propagator fixed?
basically how should we extract the pde?"): the propagator is kept in
`.eval()` and excluded from the optimizer entirely -- only `pde_head`
trains, with no competing pressure on the propagator. The mutual-vs-
detached distinction from Stage 2's ordinary continuation is moot here
(propagator receives no gradient regardless, since it isn't in the
optimizer and its parameters are set `requires_grad=False`).
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import AuxPropagatorConfig, PropagatorConfig, Stage2TrainingConfig
from ks_latent.models.propagator import AuxPropagator, LatentPropagator
from ks_latent.training.loops import train_stage2
from ks_latent.utils.seeding import set_seed


def _build(d_latent: int = 20, dropout: float = 0.1):
    prop_cfg = PropagatorConfig(
        d_latent=d_latent, hidden=16, n_blocks=1, dropout=dropout,
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


def test_freeze_propagator_requires_pde_head():
    set_seed(0)
    prop, _ = _build()
    train_seq = torch.randn(6, 20, 20)
    val_seq = torch.randn(2, 20, 20)
    train_cfg = Stage2TrainingConfig(epochs=1, batch_size=8, k_max=3, k_warmup_epochs=1)
    with pytest.raises(ValueError, match="requires a non-None pde_head"):
        train_stage2(prop, train_seq, val_seq, train_cfg, torch.device("cpu"), freeze_propagator=True)


def test_propagator_state_unchanged_pde_head_changes():
    set_seed(0)
    prop, pde_head = _build()
    prop_before = {k: v.clone() for k, v in prop.state_dict().items()}
    pde_before = {k: v.clone() for k, v in pde_head.state_dict().items()}

    train_seq = torch.randn(6, 20, 20)
    val_seq = torch.randn(2, 20, 20)
    train_cfg = Stage2TrainingConfig(
        lr=3e-2, epochs=3, batch_size=8, k_max=4, k_warmup_epochs=1,
        w_pde_distill=1.0, w_pde_rollout=0.5,
    )
    train_stage2(
        prop, train_seq, val_seq, train_cfg, torch.device("cpu"),
        pde_head=pde_head, freeze_propagator=True,
    )

    prop_after = prop.state_dict()
    pde_after = pde_head.state_dict()
    assert all(torch.equal(prop_before[k], prop_after[k]) for k in prop_before), (
        "propagator must be completely unchanged when freeze_propagator=True"
    )
    assert any(not torch.equal(pde_before[k], pde_after[k]) for k in pde_before), (
        "pde_head must actually train"
    )


def test_val_kmax_mse_constant_across_epochs():
    """A frozen, truly-eval-mode propagator's own val_kmax_mse must not
    fluctuate epoch to epoch -- catches eval_stage2_kmax's own
    unconditional propagator.train() call silently re-enabling dropout."""
    set_seed(0)
    prop, pde_head = _build(dropout=0.3)  # a large dropout would show up clearly if not truly frozen
    train_seq = torch.randn(6, 20, 20)
    val_seq = torch.randn(2, 20, 20)
    train_cfg = Stage2TrainingConfig(
        lr=3e-2, epochs=4, batch_size=8, k_max=4, k_warmup_epochs=1, w_pde_distill=1.0,
    )
    result = train_stage2(
        prop, train_seq, val_seq, train_cfg, torch.device("cpu"),
        pde_head=pde_head, freeze_propagator=True,
    )
    vals = [h["val_kmax_mse"] for h in result.train_history]
    assert all(v == vals[0] for v in vals)


def test_pde_distill_and_rollout_logged_in_history():
    set_seed(0)
    prop, pde_head = _build()
    train_seq = torch.randn(6, 20, 20)
    val_seq = torch.randn(2, 20, 20)
    train_cfg = Stage2TrainingConfig(
        lr=3e-2, epochs=2, batch_size=8, k_max=4, k_warmup_epochs=1,
        w_pde_distill=1.0, w_pde_rollout=0.5,
    )
    result = train_stage2(
        prop, train_seq, val_seq, train_cfg, torch.device("cpu"),
        pde_head=pde_head, freeze_propagator=True,
    )
    for h in result.train_history:
        assert "pde_distill" in h and "pde_rollout" in h
        assert torch.isfinite(torch.tensor(h["pde_distill"]))
        assert torch.isfinite(torch.tensor(h["pde_rollout"]))
