#!/bin/zsh
# User-directed 2026-09-09, "fine, why don't we just have a schedule that
# slowly ramps up w_pred loss" (after Section 122 -- physics_prior+etdrk4
# + correction_scale warmup + MLP inner encoder, isolating whether the
# ViT itself was the problem -- STILL collapsed by epoch 79, same
# qualitative "settle into a static spatial pattern" signature as every
# ViT-based attempt. Ruled out the ViT architecture as the primary cause:
# the same failure persists regardless of encoder family, pointing back
# to the loss function itself).
#
# NEW: `w_pred_warmup_epochs` (Stage1TrainingConfig) -- the same
# homotopy idea as `physics_prior_correction_warmup_epochs`, applied one
# level up: linearly ramps the EFFECTIVE w_pred weight from 0.0 (pure
# reconstruction) to its full value over the first N epochs, instead of
# (here: alongside) delaying just the learned correction's own growth.
# Gives the encoder pressure-free time to become a faithful
# reconstruction of the real field before ANY prediction loss -- which
# rewards contraction whenever z is imprecise, the mechanism diagnosed
# earlier today -- touches it at all.
#
# This run uses BOTH warmups together: --w-pred-warmup-epochs 40 (a real
# pure-reconstruction head start) and --physics-prior-correction-warmup-
# epochs 150 (unchanged from Sections 121/122) -- during the first ~40
# epochs there is essentially no prediction-loss gradient at all, so the
# correction's own schedule grows without real consequence until the
# encoder has had a real head start on reconstruction alone.
#
# Keeps Section 122's MLP inner encoder (--spectral-field-inner mlp) --
# already shown to train ~4x faster than the ViT with no qualitative
# downside, and ruled out as the root cause, so no reason to revert it.
#
# Verified via real smoke tests before launching (2026-09-09): trains
# cleanly (val_recon_final=0.021 after 5 epochs); a 200-step rollout on
# that checkpoint completed without diverging; Phase 2 continuation
# smoke test held val_kmax_mse~0.0144-0.0147 across 5 epochs, finite
# throughout.
#
# Otherwise identical to Section 122: K=24, N_w=64, etdrk4,
# spectral_physics_prior=True, MLP inner encoder, polynomial correction
# degree=2, max_term_order=5, standard normalization, w_shape_floor=0.1,
# no other new regularizers.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section123_spectralfield_mlp_L22_K24_Nw64_physicsprior_wpredwarmup

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE with MLP inner model (K=24, N_w=64) + REAL spectral_pde aux propagator (ETDRK4, ode_substeps=3, PHYSICS_PRIOR ACTIVE, correction_scale warmup 0.0->1.0 over 150/200 epochs, NEW w_pred warmup 0.0->1.5 over first 40 epochs, polynomial correction degree=2, max_order=4, max_term_order=5, --full-propagator), w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --spectral-field-inner mlp --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator etdrk4 --ode-substeps 3 \
  --spectral-physics-prior --physics-prior-correction-warmup-epochs 150 --w-pred-warmup-epochs 40 \
  --spectral-field-kind polynomial --spectral-poly-degree 2 --spectral-poly-max-term-order 5 \
  --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
  --w-lowpass-rollout 0.02 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Phase 2 (extend rollout): warm-started from Phase 1's own propagator, k_pred=2 -> k_max=12, w_lowpass_rollout=1.0, --amp, 80 epochs, then Gate 3/4 ==="
STAGE2_TAG="${TAG}_extended"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp --k-max 12 --k-mid 8 --k-warmup-epochs 56 --k-mid-epochs 46 --epochs 80 \
  --w-lowpass-rollout 1.0 --lowpass-rollout-power 1.0 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt
mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --dt-snap 1.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1 &
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dt-snap 1.0 --L 22.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1 &
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --L 22.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1 &
wait

echo "=== Section 123 complete ==="
