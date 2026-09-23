#!/bin/zsh
# User-directed 2026-09-10: "so it looks like it collapsed in stage 2?
# please wire that fix you mentioned into stage 2 and launch another run."
#
# Section 131 (Section 98's vit encoder/decoder + masked_mlp_expand
# propagator + w_spectrum_shape=0.1 in STAGE 1 ONLY) looked promising right
# after Stage 1 -- propagator Jacobian volume-change factor 0.68, 13/44
# singular values >= 1.0 -- but fully re-collapsed during Stage 2's own
# UNREGULARIZED 300-epoch continuation:
#
#   Stage-2 final: top 5 [1.451, 1.239, 1.115, 1.095, 1.022]
#                  bottom 5 [0.602, 0.581, 0.576, 0.538, 0.220]
#                  singular values >= 1.0: 5 of 44
#                  volume-change factor per step: 5.3e-5
#                  rollout: separation 2.70 -> 0.078, cross-sample z.std
#                  at final step: 0.059 (down from a healthy 0.96 after
#                  Stage 1 alone)
#
# Root cause: `w_spectrum_shape` was a Stage1TrainingConfig field only --
# train_stage2 had NO anti-collapse pressure of this kind at all, despite
# being the LONGER-horizon phase (k ramping 2->12 over 300 epochs) where
# it's needed most.
#
# NEW: `w_spectrum_shape`/`spectrum_shape_n_expand`/
# `spectrum_shape_expand_target`/`spectrum_shape_contract_floor`/
# `spectrum_shape_n_samples` added to `Stage2TrainingConfig` and wired into
# `train_stage2` (ks_latent/training/loops.py) -- same mechanism as Stage
# 1's copy (propagator_spectrum_shape_loss, once per epoch, outside
# autocast), evaluated directly on `propagator.step_one` using real,
# UNNOISED encoded states drawn from each batch's own windows (no rollout
# needed, unlike w_spatial/w_lowpass_rollout/w_logdet_rollout which all act
# on z_pred). Guarded by `propagator.mode == "markovian"` and
# `not freeze_propagator` (skipped entirely in the Phase-3 pde_head-only
# mode, where propagator isn't in the optimizer at all). CLI flags added
# to scripts/train_stage2_patched.py mirroring Stage 1's own. Verified via
# a real --profile smoke run of BOTH stages with the flag active, no
# errors; full unit suite re-run for regressions.
#
# Everything else UNCHANGED from Section 128/130/131: Section 98's exact
# vit encoder+decoder, Section 98's exact regularizers, masked_mlp_expand
# propagator (--prop-attn-window 3 --masked-mlp-expand-factor 3),
# --full-propagator, 200 Stage-1 epochs + 300 Stage-2 epochs at the
# Section 52/98/100/128/130/131 k12 curriculum, SAME w_spectrum_shape
# hyperparameters as Section 131 (n_expand=11, expand_target=1.5,
# contract_floor=0.7, weight=0.1) -- now applied in BOTH stages, the one
# genuinely new variable this run isolates.
#
# Phase-1-first, but this time also checking the FINAL Stage-2 checkpoint's
# full spectrum automatically (not just Stage 1's, which is all Sections
# 128-131's own [2/3] step checked) -- this is exactly the diagnostic gap
# that made Section 131's "looks fine after Stage 1" reading misleading.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section132_vitonly_dmodel56_tokenmlp_propmaskedmlpexpand_w3_x3_spectrumshape_bothstages_200ep

SPECTRUM_ARGS=(--w-spectrum-shape 0.1 --spectrum-shape-n-expand 11 --spectrum-shape-expand-target 1.5 --spectrum-shape-contract-floor 0.7 --spectrum-shape-n-samples 32)

echo "=== [1/4] Stage 1: Section 98's EXACT vit encoder+decoder + masked_mlp_expand propagator (attn_window=3, expand_factor=3) + w_spectrum_shape=0.1 (n_expand=11, expand_target=1.5, contract_floor=0.7), Section 98's EXACT regularizers, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone masked_mlp_expand --mode markovian \
  --d-model 56 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --prop-attn-window 3 --masked-mlp-expand-factor 3 \
  "${SPECTRUM_ARGS[@]}" \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.04 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.003 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

_spectrum_check() {
  local AE_PATH=$1
  local PROP_PATH=$2
  local LABEL=$3
  mamba run -n da_env python -c "
import h5py, numpy as np, torch
from torch.func import jacrev
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.diagnostics import propagator_step_jacobian_spectral_norms

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE_PATH')
prop, prop_cfg, _ = load_propagator_checkpoint('$PROP_PATH')
ae.eval(); prop.eval()

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj.shape
with torch.no_grad():
    z_all = ae.encode(traj.reshape(n*T, NX)).reshape(n, T, -1)

res = propagator_step_jacobian_spectral_norms(prop, z_all, n_samples=200, seed=0)
print('[$LABEL] top singular value: median=%.4f  p95=%.4f  min=%.4f  max=%.4f' % (
    np.median(res), np.percentile(res, 95), res.min(), res.max()
))

z0 = z_all[0, 0]
J = jacrev(lambda z: prop.step_one(z.unsqueeze(0)).squeeze(0))(z0)
sv = torch.linalg.svdvals(J)
print('[$LABEL] top 5:', sv[:5].tolist())
print('[$LABEL] bottom 5:', sv[-5:].tolist())
print('[$LABEL] singular values >= 1.0:', int((sv >= 1.0).sum()), 'out of', sv.numel())
print('[$LABEL] per-step volume-change factor:', sv.prod().item())
print('[$LABEL] sum(log(singular values)):', torch.log(sv).sum().item())

z0b = z_all[:, 0, :]
with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=60)
sep_start = (traj_roll[:, 5, :] - traj_roll[:, 0, :]).norm(dim=-1).mean().item()
sep_end = (traj_roll[:, -1, :] - traj_roll[:, -6, :]).norm(dim=-1).mean().item()
cross_sample_std_end = traj_roll[:, -1, :].std(dim=0).mean().item()
print('[$LABEL] rollout separation steps 0-5: %.4f   steps 54-59: %.4f   cross-sample z.std final step: %.4f' % (sep_start, sep_end, cross_sample_std_end))
"
}

echo "=== [2/4] FULL Jacobian spectrum check on the STAGE-1 propagator ==="
_spectrum_check "$AE" "$AUX" "stage1" > artifacts/logs/jacobiancheck_stage1_${TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_stage1_${TAG}.log

echo "=== [3/4] Stage 2: warm-started from Stage 1's own masked_mlp_expand aux, w_spectrum_shape ACTIVE THROUGHOUT this time, --amp, k_max=12, 300 epochs (Section 52/98/100/128/130/131's shared curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  "${SPECTRUM_ARGS[@]}" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [4/4] FULL Jacobian spectrum check on the FINAL STAGE-2 propagator (the exact check that caught Section 131's regression) ==="
_spectrum_check "$AE" "$STAGE2_PROP" "stage2-final" > artifacts/logs/jacobiancheck_stage2_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_stage2_${STAGE2_TAG}.log

echo "=== Section 132 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
