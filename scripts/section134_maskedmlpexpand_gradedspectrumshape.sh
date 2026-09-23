#!/bin/zsh
# User-directed 2026-09-10: "yeah build the graded version as 134. kill
# 133, it's going to collapse I fear. up the weight from .4 to .5 too."
#
# Section 133 (flat expand_target=1.1/contract_floor=0.6, weight=0.4) was
# killed mid-Stage-1 before finishing, after a robust (200-sample)
# per-rank measurement of Section 85's own real propagator showed the true
# spectrum is a smooth graded decline across ALL 44 ranks -- NOT two flat
# plateaus:
#
#   rank:    0      1      2      3      4      5      6     ...   12
#   median: 1.602  1.497  1.422  1.367  1.326  1.284  1.243  ...  0.955
#   (continuing down through the bottom 31 ranks: median 0.509, tailing
#   to 0.15-0.3 at the very bottom)
#
# Both prior flat-target attempts were therefore wrong in different
# directions: expand_target=1.5 (Sections 131-132) overshoots everything
# past about rank 5; expand_target=1.1 (Section 133) undershoots the top
# few ranks badly (rank 0 should be ~1.6) while only roughly matching the
# bottom of that same group. Neither a flat 1.5 nor a flat 1.1 across all
# 13 "expanding" ranks reflects the real system.
#
# NEW: `propagator_graded_spectrum_shape_loss`
# (ks_latent/training/losses.py) -- a per-RANK floor against a full
# reference spectrum (not a two-group split): `loss =
# mean(relu(target_spectrum - sv)^2)` across all d ranks at once.
# `ks_latent.analysis.diagnostics.propagator_step_jacobian_full_spectrum`
# (new) computes the full per-sample spectrum (not just the top entry,
# generalizing the existing D9 diagnostic); `scripts/
# compute_reference_spectrum.py` (new) samples 200 real points on a
# trusted checkpoint and saves the per-rank MEDIAN to a .npy file -- a
# robust target, not the single noisy point an earlier check this session
# used before this proper version. Run this turn against Section 85's own
# checkpoint (this project's best-trusted L=100 model, D_KY=21.76,
# n_positive=13, lambda1=0.086), producing
# artifacts/reference_spectra/section85_L100_d44.npy -- the exact numbers
# quoted above.
#
# Wired into BOTH Stage1TrainingConfig and Stage2TrainingConfig as
# `w_spectrum_shape_graded`/`spectrum_shape_graded_reference_path`/
# `spectrum_shape_graded_n_samples` -- same expensive/once-per-epoch/
# mode="markovian"-only convention as the existing (two-group)
# `w_spectrum_shape`, and independently toggleable (both may be used
# together, though this run uses ONLY the new graded mechanism, per the
# user's "build the graded version" direction -- a clean test of the new
# mechanism alone, not conflated with the old one). CLI flags added to
# both scripts/train_stage1_patched.py and scripts/train_stage2_patched.py.
# 22 new unit tests (full-spectrum helper + graded loss, including a
# reproduction of Section 130's exact collapsed-tail failure mode and a
# direct check that the full-spectrum helper's top entry matches the
# existing D9 diagnostic exactly); full suite re-run for regressions; a
# real --profile smoke run of BOTH stages with the new flag active, no
# errors.
#
# Weight raised 0.1 -> 0.4 -> 0.5 per direct user instruction this turn
# ("up the weight from .4 to .5 too") -- applied together with the graded
# retargeting, not held back for a separate isolation run.
#
# Everything else UNCHANGED from Section 128/130/131/132/133: Section
# 98's exact vit encoder+decoder, Section 98's exact regularizers,
# masked_mlp_expand propagator (--prop-attn-window 3
# --masked-mlp-expand-factor 3), --full-propagator, 200 Stage-1 epochs +
# 300 Stage-2 epochs at the Section 52/98/100/128/130-133 k12 curriculum.
#
# Phase-1-first, with the SAME two-point spectrum check as Sections
# 132-133 (Stage-1 propagator AND the final Stage-2 propagator).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section134_vitonly_dmodel56_tokenmlp_propmaskedmlpexpand_w3_x3_gradedspectrumshape_w05_200ep

REFERENCE_SPECTRUM=artifacts/reference_spectra/section85_L100_d44.npy
SPECTRUM_ARGS=(--w-spectrum-shape-graded 0.5 --spectrum-shape-graded-reference-path "$REFERENCE_SPECTRUM" --spectrum-shape-graded-n-samples 32)

echo "=== [1/4] Stage 1: Section 98's EXACT vit encoder+decoder + masked_mlp_expand propagator (attn_window=3, expand_factor=3) + NEW w_spectrum_shape_graded=0.5 (per-rank reference from Section 85's own real spectrum), Section 98's EXACT regularizers, --amp, 200 epochs ==="
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

reference = np.load('$REFERENCE_SPECTRUM')
print('[$LABEL] max abs deviation from reference spectrum:', float(np.abs(sv.detach().numpy() - reference).max()))

z0b = z_all[:, 0, :]
with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=60)
sep_start = (traj_roll[:, 5, :] - traj_roll[:, 0, :]).norm(dim=-1).mean().item()
sep_end = (traj_roll[:, -1, :] - traj_roll[:, -6, :]).norm(dim=-1).mean().item()
cross_sample_std_end = traj_roll[:, -1, :].std(dim=0).mean().item()
print('[$LABEL] rollout separation steps 0-5: %.4f   steps 54-59: %.4f   cross-sample z.std final step: %.4f' % (sep_start, sep_end, cross_sample_std_end))
"
}

echo "=== [2/4] FULL Jacobian spectrum check on the STAGE-1 propagator (against the reference spectrum) ==="
_spectrum_check "$AE" "$AUX" "stage1" > artifacts/logs/jacobiancheck_stage1_${TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_stage1_${TAG}.log

echo "=== [3/4] Stage 2: warm-started from Stage 1's own masked_mlp_expand aux, w_spectrum_shape_graded ACTIVE THROUGHOUT, --amp, k_max=12, 300 epochs (Section 52/98/100/128-133's shared curriculum) ==="
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

echo "=== Section 134 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
