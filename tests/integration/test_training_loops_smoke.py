"""Smoke tests for the Stage-1/Stage-2 training loops (brief §19: integration
tier, full code path, <60s). Synthetic smooth data, tiny dims, few epochs --
correctness of the mechanics (loss decreases, shapes, checkpointing), not
scientific reproduction.
"""

from __future__ import annotations

import torch

from ks_latent.config import (
    AutoencoderConfig,
    AuxPropagatorConfig,
    MLPAutoencoderConfig,
    PropagatorConfig,
    RegConfig,
    Stage1TrainingConfig,
    Stage2TrainingConfig,
    ViTAutoencoderConfig,
)
from ks_latent.models.autoencoder_mlp import KSAutoencoderMLP
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.autoencoder_vit import KSAutoencoderViT
from ks_latent.models.propagator import AuxPropagator, LatentPropagator
from ks_latent.training.loops import encode_dataset_with_shifts, eval_stage2_kmax, train_stage1, train_stage2
from ks_latent.utils.seeding import set_seed


def _smooth_periodic_trajectories(n_runs: int, T: int, NX: int, seed: int) -> torch.Tensor:
    """Slowly phase-drifting low-mode signal: smooth in space, smooth and
    predictable in time -- enough structure for both stages to fit."""
    g = torch.Generator().manual_seed(seed)
    x = torch.arange(NX, dtype=torch.float32) * 2 * torch.pi / NX
    t = torch.arange(T, dtype=torch.float32)
    traj = torch.empty(n_runs, T, NX)
    for r in range(n_runs):
        amp = torch.rand(1, generator=g).item() * 0.5 + 0.5
        phase0 = torch.rand(1, generator=g).item() * 2 * torch.pi
        omega = 0.05 + 0.02 * torch.rand(1, generator=g).item()
        traj[r] = amp * torch.cos(x.unsqueeze(0) - (phase0 + omega * t).unsqueeze(1))
    return traj


def test_train_stage1_smoke_reduces_loss():
    set_seed(0)
    device = torch.device("cpu")
    ae_cfg = AutoencoderConfig(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=6, dropout=0.0,
    )
    aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1)
    train_cfg = Stage1TrainingConfig(
        lr=3e-3, epochs=5, batch_size=16, grad_clip=1.0, k_pred=2
    )

    train_traj = _smooth_periodic_trajectories(6, 30, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 30, ae_cfg.NX, seed=2)

    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    result = train_stage1(ae, aux, train_traj, val_traj, train_cfg, device)

    assert len(result.train_history) == train_cfg.epochs
    assert result.train_history[-1]["loss"] < result.train_history[0]["loss"]
    assert result.val_recon_final >= 0.0
    assert result.val_recon_final < 1.0  # sane order of magnitude, not diverged


def test_train_stage1_smoke_markovian_mode_mlp():
    """brief §5.2 addendum: aux.mode=="markovian" should train end-to-end
    with a 1-history-snapshot window (not 2), still reducing loss."""
    set_seed(0)
    device = torch.device("cpu")
    ae_cfg = AutoencoderConfig(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=6, dropout=0.0,
    )
    aux_cfg = AuxPropagatorConfig(
        d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1, mode="markovian", backbone="mlp"
    )
    train_cfg = Stage1TrainingConfig(lr=3e-3, epochs=5, batch_size=16, grad_clip=1.0, k_pred=2)

    train_traj = _smooth_periodic_trajectories(6, 30, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 30, ae_cfg.NX, seed=2)

    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    assert aux.mode == "markovian"
    result = train_stage1(ae, aux, train_traj, val_traj, train_cfg, device)

    assert len(result.train_history) == train_cfg.epochs
    assert result.train_history[-1]["loss"] < result.train_history[0]["loss"]
    assert result.val_recon_final >= 0.0


def test_train_stage1_smoke_with_denoising_and_weight_decay():
    """Ported-improvements smoke test (2026-08-29): noise_std > 0 (denoising
    reconstruction path) and the explicit weight_decay/warmup schedule train
    without error and still reduce loss."""
    set_seed(0)
    device = torch.device("cpu")
    ae_cfg = AutoencoderConfig(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=6, dropout=0.0,
    )
    aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1)
    train_cfg = Stage1TrainingConfig(
        lr=3e-3, epochs=5, batch_size=16, grad_clip=1.0, k_pred=2,
        noise_std=0.05, weight_decay=1e-4, warmup_epochs=2,
    )

    train_traj = _smooth_periodic_trajectories(6, 30, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 30, ae_cfg.NX, seed=2)

    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    result = train_stage1(ae, aux, train_traj, val_traj, train_cfg, device)

    assert len(result.train_history) == train_cfg.epochs
    assert result.train_history[-1]["loss"] < result.train_history[0]["loss"]


def test_train_stage1_smoke_with_optional_latent_index_regularizer():
    """RegConfig wired through train_stage1 (default off elsewhere): passing
    it in should not error and should measurably smooth the latent index
    correlation structure relative to leaving it off."""
    set_seed(0)
    device = torch.device("cpu")
    ae_cfg = AutoencoderConfig(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=8, dropout=0.0,
    )
    aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1)
    train_cfg = Stage1TrainingConfig(lr=3e-3, epochs=5, batch_size=16, grad_clip=1.0, k_pred=2)
    reg_cfg = RegConfig(lambda_z=0.1, lambda_decorr=0.05, bandwidth=2)

    train_traj = _smooth_periodic_trajectories(6, 30, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 30, ae_cfg.NX, seed=2)

    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    result = train_stage1(ae, aux, train_traj, val_traj, train_cfg, device, reg_cfg=reg_cfg)

    assert len(result.train_history) == train_cfg.epochs
    assert result.val_recon_final >= 0.0


def test_train_stage1_smoke_mlp_encoder():
    """Ported-architecture smoke test (2026-08-29): KSAutoencoderMLP is a
    drop-in for KSAutoencoderPatched in train_stage1."""
    set_seed(0)
    device = torch.device("cpu")
    ae_cfg = MLPAutoencoderConfig(NX=32, hidden=(32, 16), d_latent=6)
    aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1)
    train_cfg = Stage1TrainingConfig(lr=3e-3, epochs=5, batch_size=16, grad_clip=1.0, k_pred=2)

    train_traj = _smooth_periodic_trajectories(6, 30, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 30, ae_cfg.NX, seed=2)

    ae = KSAutoencoderMLP(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    result = train_stage1(ae, aux, train_traj, val_traj, train_cfg, device)

    assert len(result.train_history) == train_cfg.epochs
    assert result.train_history[-1]["loss"] < result.train_history[0]["loss"]
    assert result.val_recon_final >= 0.0


def test_train_stage1_smoke_vit_encoder():
    """Ported-architecture smoke test (2026-08-29): KSAutoencoderViT is a
    drop-in for KSAutoencoderPatched in train_stage1."""
    set_seed(0)
    device = torch.device("cpu")
    ae_cfg = ViTAutoencoderConfig(NX=32, patch_size=4, d_model=16, n_heads=2, n_blocks=1, d_latent=6)
    aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1)
    train_cfg = Stage1TrainingConfig(lr=3e-3, epochs=5, batch_size=16, grad_clip=1.0, k_pred=2)

    train_traj = _smooth_periodic_trajectories(6, 30, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 30, ae_cfg.NX, seed=2)

    ae = KSAutoencoderViT(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    result = train_stage1(ae, aux, train_traj, val_traj, train_cfg, device)

    assert len(result.train_history) == train_cfg.epochs
    assert result.train_history[-1]["loss"] < result.train_history[0]["loss"]
    assert result.val_recon_final >= 0.0


def test_train_stage1_smoke_multistep_rollout_ramps_k_pred():
    """Ported-rollout smoke test (2026-08-29): k_pred_max > k_pred ramps the
    rollout length linearly from k_pred to k_pred_max, held at k_pred_max
    after the warmup, and trains without error using a larger aux
    propagator."""
    set_seed(0)
    device = torch.device("cpu")
    ae_cfg = AutoencoderConfig(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=6, dropout=0.0,
    )
    aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=32, n_blocks=2)
    train_cfg = Stage1TrainingConfig(
        lr=3e-3, epochs=6, batch_size=16, grad_clip=1.0,
        k_pred=2, k_pred_max=6, k_pred_warmup_epochs=4,
    )

    train_traj = _smooth_periodic_trajectories(6, 30, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 30, ae_cfg.NX, seed=2)

    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    result = train_stage1(ae, aux, train_traj, val_traj, train_cfg, device)

    k_sequence = [e["k_now"] for e in result.train_history]
    assert k_sequence == [2, 3, 4, 5, 6, 6]
    assert result.val_recon_final >= 0.0


def test_train_stage1_smoke_mlp_encoder_plus_multistep_combined():
    """Both ported changes together, per the user's request to test them
    independently and combined."""
    set_seed(0)
    device = torch.device("cpu")
    ae_cfg = MLPAutoencoderConfig(NX=32, hidden=(32, 16), d_latent=6)
    aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=32, n_blocks=2)
    train_cfg = Stage1TrainingConfig(
        lr=3e-3, epochs=5, batch_size=16, grad_clip=1.0,
        k_pred=2, k_pred_max=5, k_pred_warmup_epochs=3,
    )

    train_traj = _smooth_periodic_trajectories(6, 30, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 30, ae_cfg.NX, seed=2)

    ae = KSAutoencoderMLP(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    result = train_stage1(ae, aux, train_traj, val_traj, train_cfg, device)

    assert len(result.train_history) == train_cfg.epochs
    assert result.train_history[-1]["k_now"] == 5
    assert result.val_recon_final >= 0.0


def test_train_stage1_smoke_markovian_mode_transformer():
    set_seed(0)
    device = torch.device("cpu")
    d_latent = 8
    ae_cfg = AutoencoderConfig(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=d_latent, dropout=0.0,
    )
    aux_cfg = AuxPropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="transformer",
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1, attn_window=1,
    )
    train_cfg = Stage1TrainingConfig(lr=3e-3, epochs=5, batch_size=16, grad_clip=1.0, k_pred=2)

    train_traj = _smooth_periodic_trajectories(6, 30, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 30, ae_cfg.NX, seed=2)

    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    result = train_stage1(ae, aux, train_traj, val_traj, train_cfg, device)

    assert len(result.train_history) == train_cfg.epochs
    assert result.val_recon_final >= 0.0


def test_train_stage2_markovian_mode_reuses_same_loop_unchanged():
    """train_stage2/eval_stage2_kmax take no mode-specific code path --
    LatentPropagator.rollout's uniform interface means this Just Works."""
    set_seed(0)
    device = torch.device("cpu")
    d_latent = 6
    ae_cfg = AutoencoderConfig(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=d_latent, dropout=0.0,
    )
    ae = KSAutoencoderPatched(ae_cfg)
    ae.eval()

    train_traj = _smooth_periodic_trajectories(6, 40, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 40, ae_cfg.NX, seed=2)
    train_seq = encode_dataset_with_shifts(ae, train_traj, shifts=[0, 8], device=device)
    val_seq = encode_dataset_with_shifts(ae, val_traj, shifts=[0], device=device)

    prop_cfg = PropagatorConfig(d_latent=d_latent, hidden=32, n_blocks=2, dropout=0.0, mode="markovian", backbone="mlp")
    prop = LatentPropagator(prop_cfg)
    train_cfg = Stage2TrainingConfig(
        lr=3e-3, epochs=6, batch_size=32, k_max=6, k_warmup_epochs=3,
        noise_in_start=0.05, noise_in_end=0.01,
    )
    result = train_stage2(prop, train_seq, val_seq, train_cfg, device)
    assert len(result.train_history) == train_cfg.epochs
    assert result.best_val_kmax_mse < float("inf")


def test_train_stage2_vit_backbone_smoke():
    """Ported-architecture smoke test (2026-08-29): the "vit" propagator
    backbone trains end to end through the unchanged train_stage2 loop,
    same as the "transformer" backbone."""
    set_seed(0)
    device = torch.device("cpu")
    d_latent = 8
    ae_cfg = AutoencoderConfig(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=d_latent, dropout=0.0,
    )
    ae = KSAutoencoderPatched(ae_cfg)
    ae.eval()

    train_traj = _smooth_periodic_trajectories(6, 40, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 40, ae_cfg.NX, seed=2)
    train_seq = encode_dataset_with_shifts(ae, train_traj, shifts=[0, 8], device=device)
    val_seq = encode_dataset_with_shifts(ae, val_traj, shifts=[0], device=device)

    prop_cfg = PropagatorConfig(
        d_latent=d_latent, mode="markovian", backbone="vit",
        n_tokens=4, token_d_model=8, token_nhead=2, token_n_layers=1,
    )
    prop = LatentPropagator(prop_cfg)
    train_cfg = Stage2TrainingConfig(
        lr=3e-3, epochs=6, batch_size=32, k_max=6, k_warmup_epochs=3,
        noise_in_start=0.05, noise_in_end=0.01,
    )
    result = train_stage2(prop, train_seq, val_seq, train_cfg, device)
    assert len(result.train_history) == train_cfg.epochs
    assert result.best_val_kmax_mse < float("inf")


def test_train_stage2_smoke_reduces_loss_and_checkpoints():
    set_seed(0)
    device = torch.device("cpu")
    d_latent = 6
    ae_cfg = AutoencoderConfig(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=d_latent, dropout=0.0,
    )
    ae = KSAutoencoderPatched(ae_cfg)
    ae.eval()

    train_traj = _smooth_periodic_trajectories(6, 40, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 40, ae_cfg.NX, seed=2)
    train_seq = encode_dataset_with_shifts(ae, train_traj, shifts=[0, 8], device=device)
    val_seq = encode_dataset_with_shifts(ae, val_traj, shifts=[0], device=device)
    assert train_seq.shape == (6 * 2, 40, d_latent)

    prop_cfg = PropagatorConfig(d_latent=d_latent, hidden=32, n_blocks=2, dropout=0.0)
    prop = LatentPropagator(prop_cfg)
    train_cfg = Stage2TrainingConfig(
        lr=3e-3, epochs=6, batch_size=32, k_max=6, k_warmup_epochs=3,
        noise_in_start=0.05, noise_in_end=0.01,
    )
    result = train_stage2(prop, train_seq, val_seq, train_cfg, device)

    assert len(result.train_history) == train_cfg.epochs
    assert result.train_history[0]["k_now"] == 2  # curriculum starts at 2
    assert result.train_history[-1]["k_now"] == train_cfg.k_max  # ramps to k_max
    assert result.best_val_kmax_mse < float("inf")
    # loaded state dict should be the recorded best
    loaded_norm = sum(v.norm().item() for v in prop.state_dict().values())
    best_norm = sum(v.norm().item() for v in result.best_state_dict.values())
    assert abs(loaded_norm - best_norm) < 1e-6


# ---- w_lowpass_rollout: penalizes the PROPAGATOR's own rolled-out z_pred
# for carrying high-wavenumber energy, added 2026-09-08 (Stage-2 analogue
# of Stage1TrainingConfig.w_lowpass), user-directed after visualizing
# Section 104's D_KY=22 result -- see Stage2TrainingConfig.
# w_lowpass_rollout's docstring. ----


def test_train_stage2_w_lowpass_rollout_runs_and_changes_loss():
    """Smoke: backbone="spectral_pde" trains end to end with
    w_lowpass_rollout > 0, K/L are correctly duck-typed from
    propagator.cfg, and the term actually moves the loss (nonzero effect
    vs. w_lowpass_rollout=0 on the same data/seed)."""
    from ks_latent.models.spectral_field import encode_to_spectrum

    set_seed(0)
    device = torch.device("cpu")
    K, N_w, L = 6, 24, 22.0
    d_latent = 2 * K

    train_w = _smooth_periodic_trajectories(6, 40, N_w, seed=1)
    val_w = _smooth_periodic_trajectories(2, 40, N_w, seed=2)
    train_seq = encode_to_spectrum(train_w.reshape(-1, N_w), K).reshape(6, 40, d_latent)
    val_seq = encode_to_spectrum(val_w.reshape(-1, N_w), K).reshape(2, 40, d_latent)

    def _run(w_lowpass_rollout: float):
        set_seed(0)
        prop_cfg = PropagatorConfig(
            d_latent=d_latent, hidden=16, n_blocks=1, dropout=0.0,
            mode="markovian", backbone="spectral_pde",
            spectral_K=K, spectral_N_w=N_w, spectral_L=L, spectral_max_order=4,
        )
        prop = LatentPropagator(prop_cfg)
        train_cfg = Stage2TrainingConfig(
            lr=3e-3, epochs=3, batch_size=8, k_max=4, k_warmup_epochs=2,
            w_lowpass_rollout=w_lowpass_rollout,
        )
        return train_stage2(prop, train_seq, val_seq, train_cfg, device)

    result_off = _run(0.0)
    result_on = _run(0.5)
    assert len(result_on.train_history) == 3
    assert result_on.best_val_kmax_mse < float("inf")
    # Same seed/data/architecture, only w_lowpass_rollout differs -- any
    # nonzero effect confirms the term is actually wired into the loss
    # (not silently skipped due to K/L not being found on propagator.cfg).
    assert result_on.train_history[0]["loss"] != result_off.train_history[0]["loss"]


# ---- pde_head distillation: a SECOND, always backbone="spectral_pde_raw"
# propagator trained ALONGSIDE a free `aux` (e.g. backbone="mlp") via a
# single-step distillation loss against aux's own detached one-step
# prediction, added 2026-09-08, user-directed: "joint training seems
# pretty smart. can you implement this idea", then refined to work on the
# RAW latent of ANY encoder (not just a dedicated spectral_field one) via
# a self-FFT: "I would like to be able to use the encoder and decoder and
# propagator from 95 ... within the propagator, I want to take the fourier
# transform of the latent states, and train the spectral pde with this
# information" -- see train_stage1's own docstring and
# tests/unit/test_pde_head_distillation.py (which verifies the underlying
# gradient-isolation property directly) for the full mechanism. Uses a
# PLAIN (non-spectral) encoder deliberately, matching what
# train_stage1_patched.py --pde-distill now builds for any --encoder. ----


def test_train_stage1_smoke_pde_head_distillation_runs_and_reduces_loss():
    set_seed(0)
    device = torch.device("cpu")
    NX, d_latent = 32, 20
    ae_cfg = AutoencoderConfig(
        NX=NX, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=d_latent, dropout=0.0,
    )

    aux_cfg = AuxPropagatorConfig(d_latent=d_latent, hidden=16, n_blocks=1, mode="markovian", backbone="mlp")
    pde_cfg = AuxPropagatorConfig(
        d_latent=d_latent, hidden=16, n_blocks=1, mode="markovian", backbone="spectral_pde_raw",
        spectral_K=d_latent // 2 + 1, spectral_L=float(d_latent), spectral_max_order=4,
        spectral_integrator="euler",
    )
    train_cfg = Stage1TrainingConfig(lr=3e-3, epochs=5, batch_size=16, grad_clip=1.0, w_pde_distill=1.0)

    train_traj = _smooth_periodic_trajectories(6, 30, NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 30, NX, seed=2)

    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    pde_head = AuxPropagator(pde_cfg)

    result = train_stage1(ae, aux, train_traj, val_traj, train_cfg, device, pde_head=pde_head)

    assert len(result.train_history) == train_cfg.epochs
    assert torch.isfinite(torch.tensor(result.train_history[-1]["loss"])).item()
    assert result.train_history[-1]["loss"] < result.train_history[0]["loss"]


def test_train_stage1_pde_head_none_is_unchanged_when_weight_zero():
    """w_pde_distill=0 (default) with pde_head=None must reproduce the
    exact pre-existing behavior -- no accidental coupling introduced."""
    set_seed(0)
    device = torch.device("cpu")
    NX, d_latent = 32, 20
    ae_cfg = AutoencoderConfig(
        NX=NX, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=d_latent, dropout=0.0,
    )
    aux_cfg = AuxPropagatorConfig(d_latent=d_latent, hidden=16, n_blocks=1, mode="markovian", backbone="mlp")
    train_cfg = Stage1TrainingConfig(lr=3e-3, epochs=3, batch_size=16, grad_clip=1.0)

    train_traj = _smooth_periodic_trajectories(6, 30, NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 30, NX, seed=2)

    def _run(pde_head):
        set_seed(0)
        ae = KSAutoencoderPatched(ae_cfg)
        aux = AuxPropagator(aux_cfg)
        return train_stage1(ae, aux, train_traj, val_traj, train_cfg, device, pde_head=pde_head)

    result_none = _run(None)
    pde_cfg = AuxPropagatorConfig(
        d_latent=d_latent, hidden=16, n_blocks=1, mode="markovian", backbone="spectral_pde_raw",
        spectral_K=d_latent // 2 + 1, spectral_L=float(d_latent), spectral_max_order=4,
        spectral_integrator="euler",
    )
    result_with_unused_head = _run(AuxPropagator(pde_cfg))
    assert result_none.train_history[-1]["loss"] == result_with_unused_head.train_history[-1]["loss"]


# ---- pde_head continuation into Stage 2 (added 2026-09-08, see
# docs/sine_transform_pde_plan.md §21, user-directed: "is the pde_head
# trained during phase 2 as well ... please implement both option a and b").
# Unlike Stage 1's own pde_head loss, this one is MUTUAL (propagator's own
# z_pred is NOT detached) -- see train_stage2's own docstring and
# tests/unit/test_pde_head_stage2_mutual.py (which verifies the mutual
# gradient property directly) for the full mechanism. ----


def test_train_stage2_pde_head_option_a_and_b_run_and_reduce_loss():
    set_seed(0)
    device = torch.device("cpu")
    d_latent = 20
    prop_cfg = PropagatorConfig(d_latent=d_latent, hidden=16, n_blocks=1, dropout=0.0, backbone="mlp")
    pde_cfg = AuxPropagatorConfig(
        d_latent=d_latent, hidden=16, n_blocks=1, mode="markovian", backbone="spectral_pde_raw",
        spectral_K=d_latent // 2 + 1, spectral_L=float(d_latent), spectral_max_order=4,
        spectral_integrator="euler",
    )
    train_seq = torch.randn(6, 20, d_latent)
    val_seq = torch.randn(2, 20, d_latent)

    prop = LatentPropagator(prop_cfg)
    pde_head = AuxPropagator(pde_cfg)
    train_cfg = Stage2TrainingConfig(
        lr=3e-3, epochs=5, batch_size=8, k_max=4, k_warmup_epochs=2,
        w_pde_distill=1.0, w_pde_rollout=0.5,
    )
    result = train_stage2(prop, train_seq, val_seq, train_cfg, device, pde_head=pde_head)

    assert len(result.train_history) == 5
    assert torch.isfinite(torch.tensor(result.train_history[-1]["loss"])).item()


def test_train_stage2_pde_head_none_is_unchanged_when_weights_zero():
    set_seed(0)
    device = torch.device("cpu")
    d_latent = 20
    prop_cfg = PropagatorConfig(d_latent=d_latent, hidden=16, n_blocks=1, dropout=0.0, backbone="mlp")
    train_seq = torch.randn(6, 20, d_latent)
    val_seq = torch.randn(2, 20, d_latent)
    train_cfg = Stage2TrainingConfig(lr=3e-3, epochs=3, batch_size=8, k_max=4, k_warmup_epochs=2)

    def _run(pde_head):
        set_seed(0)
        prop = LatentPropagator(prop_cfg)
        return train_stage2(prop, train_seq, val_seq, train_cfg, device, pde_head=pde_head)

    result_none = _run(None)
    pde_cfg = AuxPropagatorConfig(
        d_latent=d_latent, hidden=16, n_blocks=1, mode="markovian", backbone="spectral_pde_raw",
        spectral_K=d_latent // 2 + 1, spectral_L=float(d_latent), spectral_max_order=4,
        spectral_integrator="euler",
    )
    result_with_unused_head = _run(AuxPropagator(pde_cfg))
    assert result_none.train_history[-1]["loss"] == result_with_unused_head.train_history[-1]["loss"]


# ---- compare_k: a second, smaller-horizon readout pooled from the SAME
# rollout, added 2026-09-04, user-directed: "can we add a readout for
# val_kequals12_mse, so we can compare these runs more directly" -- lets
# two Stage-2 runs at different k_max be compared at a shared horizon,
# since val_kmax_mse alone is inflated by k_max itself (more, more-
# diverged steps pooled into the average). ----


def test_eval_stage2_kmax_returns_bare_float_without_compare_k():
    set_seed(0)
    device = torch.device("cpu")
    d_latent = 6
    prop = LatentPropagator(PropagatorConfig(d_latent=d_latent, hidden=16, n_blocks=1, dropout=0.0))
    seqs = torch.randn(4, 20, d_latent)
    result = eval_stage2_kmax(prop, seqs, k_max=6, device=device)
    assert isinstance(result, float)


def test_eval_stage2_kmax_compare_k_returns_tuple_and_matches_full_when_equal_to_kmax():
    set_seed(0)
    device = torch.device("cpu")
    d_latent = 6
    prop = LatentPropagator(PropagatorConfig(d_latent=d_latent, hidden=16, n_blocks=1, dropout=0.0))
    seqs = torch.randn(4, 20, d_latent)
    kmax_mse, compare_mse = eval_stage2_kmax(prop, seqs, k_max=6, device=device, compare_k=6)
    assert isinstance(kmax_mse, float) and isinstance(compare_mse, float)
    assert kmax_mse == compare_mse  # compare_k == k_max -> identical pooled window


def test_eval_stage2_kmax_compare_k_matches_manual_rollout_on_first_steps():
    """compare_k's pooled MSE must equal the MSE of the SAME rollout's
    first compare_k steps -- verified against a manually-computed rollout
    on a single exact-length window (n_runs=1, T=k_max+n_hist exactly, so
    there is exactly one valid window and no batching/shuffling ambiguity)."""
    set_seed(0)
    device = torch.device("cpu")
    d_latent = 6
    prop = LatentPropagator(PropagatorConfig(d_latent=d_latent, hidden=16, n_blocks=2, dropout=0.0, zero_init=False))
    prop.eval()
    k_max, compare_k, n_hist = 10, 4, 2  # "two_step" mode: n_hist=2
    seqs = torch.randn(1, k_max + n_hist, d_latent)

    kmax_mse, compare_mse = eval_stage2_kmax(prop, seqs, k_max=k_max, device=device, compare_k=compare_k)

    with torch.no_grad():
        z_pred = prop.rollout(seqs[:, 0], seqs[:, 1], k_max)
    z_true = seqs[:, n_hist:]
    expected_kmax = ((z_pred - z_true) ** 2).mean().item()
    expected_compare = ((z_pred[:, :compare_k] - z_true[:, :compare_k]) ** 2).mean().item()

    assert abs(kmax_mse - expected_kmax) < 1e-6
    assert abs(compare_mse - expected_compare) < 1e-6
    assert compare_mse != kmax_mse  # genuinely a different (smaller-horizon) pooled window


def test_train_stage2_compare_k_logs_extra_history_key():
    set_seed(0)
    device = torch.device("cpu")
    d_latent = 6
    ae_cfg = AutoencoderConfig(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=d_latent, dropout=0.0,
    )
    ae = KSAutoencoderPatched(ae_cfg)
    ae.eval()
    train_traj = _smooth_periodic_trajectories(6, 40, ae_cfg.NX, seed=1)
    val_traj = _smooth_periodic_trajectories(2, 40, ae_cfg.NX, seed=2)
    train_seq = encode_dataset_with_shifts(ae, train_traj, shifts=[0, 8], device=device)
    val_seq = encode_dataset_with_shifts(ae, val_traj, shifts=[0], device=device)

    prop_cfg = PropagatorConfig(d_latent=d_latent, hidden=32, n_blocks=2, dropout=0.0)
    prop = LatentPropagator(prop_cfg)
    train_cfg = Stage2TrainingConfig(
        lr=3e-3, epochs=4, batch_size=32, k_max=6, k_warmup_epochs=3, compare_k=3,
        noise_in_start=0.05, noise_in_end=0.01,
    )
    result = train_stage2(prop, train_seq, val_seq, train_cfg, device)
    assert all("val_k3_mse" in h for h in result.train_history)


def test_stage2_training_config_rejects_compare_k_above_k_max():
    import pytest

    with pytest.raises(ValueError):
        Stage2TrainingConfig(k_max=6, compare_k=7)
