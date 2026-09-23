"""Tests for the linear regularizer-weight decay added to `train_stage1`
(2026-09-02, user-directed: "initially having regularizer parameters high
and then decaying them overtime to increase prediction performance") --
`Stage1TrainingConfig.w_var_end`/`w_logdet_end`/`w_spatial_end` and
`ks_latent.training.loops.linear_decay`.
"""

from __future__ import annotations

import torch

from ks_latent.config import AutoencoderConfig, AuxPropagatorConfig, Stage1TrainingConfig
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.propagator import AuxPropagator
from ks_latent.training.loops import linear_decay, train_stage1
from ks_latent.utils.seeding import set_seed


def test_linear_decay_endpoints_and_midpoint():
    assert abs(linear_decay(0, 10, 0.08, 0.004) - 0.08) < 1e-9
    assert abs(linear_decay(9, 10, 0.08, 0.004) - 0.004) < 1e-9
    mid = linear_decay(4, 9, 0.08, 0.004)  # epoch 4 of 9 (last index 8) -> frac 0.5
    assert abs(mid - 0.042) < 1e-9


def test_linear_decay_single_epoch_returns_end():
    assert abs(linear_decay(0, 1, 0.08, 0.004) - 0.004) < 1e-9


def test_linear_decay_holds_end_value_past_final_epoch():
    # Defensive: an out-of-range epoch (shouldn't occur from train_stage1's
    # own loop, which never calls past cfg.epochs - 1) still clamps to `end`.
    assert abs(linear_decay(50, 10, 0.08, 0.004) - 0.004) < 1e-9


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
    return ae, aux, traj


def test_w_var_end_decays_linearly_in_history():
    set_seed(0)
    ae, aux, traj = _tiny_setup()
    cfg = Stage1TrainingConfig(
        lr=1e-3, epochs=6, batch_size=8, k_pred=1, w_var=0.05, w_var_end=0.01
    )
    result = train_stage1(ae, aux, traj, traj, cfg, torch.device("cpu"))
    w_vars = [h["w_var"] for h in result.train_history]
    assert w_vars[0] == 0.05
    assert abs(w_vars[-1] - 0.01) < 1e-9
    # Monotonically non-increasing (a pure linear decay from high to low).
    assert all(a >= b - 1e-12 for a, b in zip(w_vars, w_vars[1:]))


def test_no_decay_when_end_is_none_omits_history_keys():
    set_seed(0)
    ae, aux, traj = _tiny_setup()
    cfg = Stage1TrainingConfig(lr=1e-3, epochs=3, batch_size=8, k_pred=1, w_var=0.05)
    result = train_stage1(ae, aux, traj, traj, cfg, torch.device("cpu"))
    assert "w_var" not in result.train_history[0]


def test_w_logdet_and_w_spatial_end_decay_independently():
    set_seed(0)
    ae, aux, traj = _tiny_setup()
    cfg = Stage1TrainingConfig(
        lr=1e-3, epochs=4, batch_size=8, k_pred=1,
        w_logdet=0.02, w_logdet_end=0.0005,
        w_spatial=0.08, w_spatial_end=0.004, spatial_bandwidth=2.0,
    )
    result = train_stage1(ae, aux, traj, traj, cfg, torch.device("cpu"))
    assert result.train_history[0]["w_logdet"] == 0.02
    assert abs(result.train_history[-1]["w_logdet"] - 0.0005) < 1e-9
    assert result.train_history[0]["w_spatial"] == 0.08
    assert abs(result.train_history[-1]["w_spatial"] - 0.004) < 1e-9
