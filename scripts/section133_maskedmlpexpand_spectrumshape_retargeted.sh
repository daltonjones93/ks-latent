#!/bin/zsh
# User-directed 2026-09-10. Section 132 (w_spectrum_shape active in BOTH
# stages, n_expand=11/expand_target=1.5/contract_floor=0.7) was killed
# before finishing, after a direct ground-truth comparison showed the
# TARGETS themselves were miscalibrated, not (necessarily) the weight.
#
# Computed this turn via this project's own already-validated Benettin/QR
# Lyapunov machinery (ks_latent.analysis.lyapunov.
# lyapunov_spectrum_latent_propagator) on Section 85's checkpoint -- this
# project's best-trusted L=100 model (D_KY=21.76, n_positive=13,
# lambda1=0.086, squarely inside the established benchmark band):
#
#   full spectrum converted to one-step multipliers exp(lambda_i * dt_snap=1.0):
#     top 5 multipliers:    [1.090, 1.086, 1.085, 1.068, 1.066]
#     bottom 5 multipliers: [0.631, 0.630, 0.623, 0.619, 0.604]
#     count >= 1.0: 13 of 44
#
# This is a genuinely different, and more correct, reference than what
# Sections 130-132's expand_target=1.5 was calibrated against (Section
# 85's own D9 diagnostic MEDIAN SINGULAR VALUE, ~1.6) -- that's an
# INSTANTANEOUS quantity (the single most-stretched direction at one
# sampled instant, maximized over all directions), not the ASYMPTOTIC
# time-averaged growth rate along the aligned Lyapunov direction that
# exp(lambda_i) represents. The two are related but not the same, and the
# instantaneous one runs meaningfully hotter. Asking the propagator's top
# singular values to reach 1.5 was asking for ~35-40% more per-step
# stretching than the real attractor's own dominant direction actually
# has, which may itself have been fighting training rather than helping
# it. contract_floor=0.7 was also slightly stricter than reality (true
# bottom directions sit at 0.60-0.63).
#
# THIS SECTION retargets both, plus n_expand per user direction ("include
# 13 expanding multipliers" -- matching the real measured n_positive=13,
# not the +-2-band default of 11 used previously):
#
#   spectrum_shape_n_expand:       11 -> 13
#   spectrum_shape_expand_target: 1.5 -> 1.1  (~matches the real top multiplier ~1.09)
#   spectrum_shape_contract_floor: 0.7 -> 0.6  (~matches the real bottom multiplier ~0.60-0.63)
#
# w_spectrum_shape weight raised 0.1 -> 0.4 per direct user instruction
# (2026-09-10, "actually I want the weight to be .4 not .1"), applied
# together with the retargeting this time (not held back for a separate
# isolation run) -- n_samples (32) unchanged.
#
# Applied in BOTH stages (Section 132's own fix, still needed -- see that
# script's header for why Stage-1-only was insufficient: a checkpoint
# healthy right after Stage 1 fully re-collapsed during Stage 2's own
# unregularized continuation).
#
# Everything else UNCHANGED from Section 128/130/131/132: Section 98's
# exact vit encoder+decoder, Section 98's exact regularizers,
# masked_mlp_expand propagator (--prop-attn-window 3
# --masked-mlp-expand-factor 3), --full-propagator, 200 Stage-1 epochs +
# 300 Stage-2 epochs at the Section 52/98/100/128/130/131/132 k12
# curriculum.
#
# Phase-1-first, with the SAME two-point spectrum check as Section 132
# (Stage-1 propagator AND the final Stage-2 propagator, not just Stage 1's
# own -- exactly the diagnostic gap that made Section 131 look fine when
# it wasn't).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section133_vitonly_dmodel56_tokenmlp_propmaskedmlpexpand_w3_x3_spectrumshape_retargeted_200ep

SPECTRUM_ARGS=(--w-spectrum-shape 0.4 --spectrum-shape-n-expand 13 --spectrum-shape-expand-target 1.1 --spectrum-shape-contract-floor 0.6 --spectrum-shape-n-samples 32)

echo "=== [1/4] Stage 1: Section 98's EXACT vit encoder+decoder + masked_mlp_expand propagator (attn_window=3, expand_factor=3) + RETARGETED w_spectrum_shape=0.4 (n_expand=13, expand_target=1.1, contract_floor=0.6), Section 98's EXACT regularizers, --amp, 200 epochs ==="
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

echo "=== [3/4] Stage 2: warm-started from Stage 1's own masked_mlp_expand aux, RETARGETED w_spectrum_shape ACTIVE THROUGHOUT, --amp, k_max=12, 300 epochs (Section 52/98/100/128/130/131/132's shared curriculum) ==="
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

echo "=== [4/4] FULL Jacobian spectrum check on the FINAL STAGE-2 propagator ==="
_spectrum_check "$AE" "$STAGE2_PROP" "stage2-final" > artifacts/logs/jacobiancheck_stage2_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_stage2_${STAGE2_TAG}.log

echo "=== Section 133 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
