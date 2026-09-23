"""Tests for RegConfig.start_epoch (added 2026-09-02, user-directed:
"only add it after the first 30 epochs") -- delayed activation of the
banded latent-index-smoothness regularizer in `train_stage1`.
"""

from __future__ import annotations

import torch

from ks_latent.config import AutoencoderConfig, AuxPropagatorConfig, RegConfig, Stage1TrainingConfig
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.propagator import AuxPropagator
from ks_latent.training.loops import train_stage1
from ks_latent.utils.seeding import set_seed


def _tiny_setup():
    ae_cfg = AutoencoderConfig(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=6, dropout=0.0,
    )
    aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1)
    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    x = torch.arange(ae_cfg.NX, dtype=torch.float32) * 2 * torch.pi / ae_cfg.NX
    t = torch.arange(20, dtype=torch.float32)
    traj = torch.cos(x.unsqueeze(0) - 0.05 * t.unsqueeze(1)).unsqueeze(0).repeat(4, 1, 1)
    return ae_cfg, aux_cfg, traj


def _first_epoch_loss(reg_cfg, seed=0):
    set_seed(seed)
    ae_cfg, aux_cfg, traj = _tiny_setup()
    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    cfg = Stage1TrainingConfig(lr=1e-3, epochs=3, batch_size=8, k_pred=1)
    result = train_stage1(ae, aux, traj, traj, cfg, torch.device("cpu"), reg_cfg=reg_cfg)
    return [h["loss"] for h in result.train_history]


def test_start_epoch_zero_matches_default_active_from_epoch_zero():
    active_from_zero = RegConfig(lambda_z=0.5, bandwidth=2, start_epoch=0)
    never_reaches = RegConfig(lambda_z=0.5, bandwidth=2, start_epoch=100)
    losses_active = _first_epoch_loss(active_from_zero)
    losses_inactive = _first_epoch_loss(never_reaches)
    # Same seed, same architecture/data -- the only difference is whether
    # the (large) lambda_z penalty is added to the loss at all. Every epoch
    # should differ once the penalty is actually contributing.
    assert all(a > b for a, b in zip(losses_active, losses_inactive))


def test_start_epoch_delays_penalty_contribution():
    set_seed(0)
    ae_cfg, aux_cfg, traj = _tiny_setup()
    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    reg_cfg = RegConfig(lambda_z=0.5, bandwidth=2, start_epoch=2)
    cfg = Stage1TrainingConfig(lr=0.0, epochs=3, batch_size=8, k_pred=1)  # lr=0: weights frozen
    result = train_stage1(ae, aux, traj, traj, cfg, torch.device("cpu"), reg_cfg=reg_cfg)
    losses = [h["loss"] for h in result.train_history]
    # With frozen weights, epochs 0-1 (< start_epoch) should be close (only
    # source of variation is the random per-epoch cyclic shift augmentation
    # -- no penalty contributes yet either way); epoch 2 (>= start_epoch)
    # must jump up sharply once the (positive) lambda_z penalty activates,
    # well beyond that epoch-to-epoch augmentation noise.
    pre_activation_diff = abs(losses[0] - losses[1])
    activation_jump = losses[2] - losses[1]
    assert activation_jump > 10 * pre_activation_diff
    assert activation_jump > 0


def test_reg_config_start_epoch_defaults_to_zero():
    assert RegConfig().start_epoch == 0
