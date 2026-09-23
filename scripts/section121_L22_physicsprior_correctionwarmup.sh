#!/bin/zsh
# User-directed 2026-09-09, "how do we preserve the chaotic structure and
# nudge it in the direction we want? could we progressively add systems
# we know are chaotic? ... sort of a dynamic system gradient descent"
# (after Section 120 -- physics_prior+etdrk4, no schedule -- showed
# real spatiotemporal structure at zero training, but VISIBLY DAMPED that
# structure within the first ~20 of 200 epochs: the rollout decayed from
# persistent, full-amplitude traveling-wave motion to a temporally-static,
# low-amplitude pattern).
#
# Diagnosis: short-horizon MSE prediction loss (--w-pred) systematically
# rewards contraction whenever the encoder's z is imprecise (always true
# early in joint training) -- a damped system's errors shrink regardless
# of input quality, while a genuinely chaotic system's errors grow
# regardless of model quality. Even starting from the EXACT KS equation,
# gradient descent on this loss has a direct incentive to learn a
# correction that damps the built-in chaos.
#
# NEW: `physics_prior_correction_warmup_epochs` (Stage1TrainingConfig,
# ks_latent.models.propagator._SpectralPDEDeltaBody.correction_scale) --
# a homotopy/continuation schedule directly implementing the user's idea:
# `field(z) = exact_KS(z) + correction_scale(epoch) * learned_correction(z)`,
# with correction_scale ramping linearly 0.0 -> 1.0 over the first N
# epochs (held at 1.0 after). At correction_scale=0, the propagator is
# EXACTLY the true KS equation -- genuinely chaotic, completely untouched
# by the learned correction -- giving the encoder a long runway to
# converge to an accurate representation UNDER REAL chaotic dynamics
# before the learned correction has enough room to find the "damp
# everything" shortcut.
#
# Verified directly (2026-09-09) before launching: at correction_scale
# forced to exactly 0.0 vs 1.0 on the same randomly-initialized model,
# output differs (confirms the mechanism actually gates the learned
# contribution) and both stay finite. A real smoke run (3 epochs,
# warmup=150 so correction_scale~0.01-0.03 throughout the smoke test)
# trained cleanly (val_recon_final=0.028) and a 200-step rollout on that
# checkpoint stayed close to the pure-physics baseline's own behavior (as
# expected at this near-zero correction_scale) with no divergence; Phase 2
# continuation smoke test held val_kmax_mse~0.097-0.098 across 5 epochs.
# Full test suite (238 propagator/config/loops tests) passes, no
# regressions.
#
# Otherwise identical to Section 120: K=24, N_w=64, etdrk4,
# spectral_physics_prior=True, polynomial correction degree=2,
# max_term_order=5, standard normalization, w_shape_floor=0.1 (Section
# 104's own baseline), no other new regularizers -- isolating the
# warmup schedule's own effect cleanly against Section 120's own
# (damped) result.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section121_spectralfield_L22_K24_Nw64_physicsprior_correctionwarmup

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=24, N_w=64) + REAL spectral_pde aux propagator (ETDRK4, ode_substeps=3, PHYSICS_PRIOR ACTIVE, NEW correction warmup: correction_scale ramps 0.0->1.0 over first 150/200 epochs, polynomial correction degree=2, max_order=4, max_term_order=5, --full-propagator), w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator etdrk4 --ode-substeps 3 \
  --spectral-physics-prior --physics-prior-correction-warmup-epochs 150 \
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

echo "=== [2/3] Phase 2 (extend rollout): warm-started from Phase 1's own propagator (correction_scale already at 1.0 by the end of Phase 1's 150-epoch warmup), k_pred=2 -> k_max=12, w_lowpass_rollout=1.0, --amp, 80 epochs, then Gate 3/4 ==="
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

echo "=== Section 121 complete ==="
