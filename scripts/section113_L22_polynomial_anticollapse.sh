#!/bin/zsh
# User-directed 2026-09-09, "Section 113": "any ideas to make it not
# collapse? try these in 113. launch this with your best guess now."
#
# Section 112's own recipe (K=24, N_w=64, L=22, euler, ode_substeps=3,
# polynomial field_kind, degree=2, max_order=4, max_term_order=5,
# normalization fix active, w_pred=1.5, w_lowpass=0.0012,
# w_lowpass_rollout=0.02 Phase1/1.0 Phase2) COLLAPSED -- confirmed via
# visualize_rollout.py's Hovmoller plots (latent AND physical space):
# every latent channel snaps to a constant at t=0 and never changes for
# 200 steps, val_kmax_mse=0.0015 (vs. Section 98's own healthy 0.033).
#
# Likely cause: today's two stability fixes (fixed-scale normalization,
# combined-order term truncation) both directly SHRINK the model's dynamic
# range/hypothesis space -- necessary to stop the earlier NaN blowup, but
# plausibly also made "contract to a point" the cheapest solution again --
# the same H-PROP mechanism this whole project keeps re-encountering.
# Section 104's own ORIGINAL (unnormalized, unrestricted, MLP field_kind)
# run at the same w_shape_floor=0.1 did NOT collapse -- only switching to
# the (now-stabilized, but also now-constrained) polynomial form did.
#
# TWO changes, both using ALREADY-BUILT, directly-targeted anti-collapse
# tools this project built specifically for this failure mode, neither
# used anywhere in the Section 104-112 lineage:
#
#   1. --w-shape-floor 0.1 -> 0.3 (Phase 1, encoder-side anti-collapse,
#      escalated to Section 105's own precedent value for "didn't work at
#      the gentler setting, make it count").
#
#   2. Phase 2 NOW ALSO carries:
#      --w-varmatch 0.1 --w-varmatch-adaptive: penalizes the PROPAGATOR's
#      own rolled-out z_pred for having LOW cross-initial-condition
#      variance -- literally the tool this project built after finding
#      "every tested initial condition converged to the SAME single fixed
#      point" (Stage2TrainingConfig.w_varmatch's own docstring/history) --
#      exactly the failure mode Section 112 just reproduced. Adaptive
#      target (real per-channel variance from data, not a flat 1) avoids
#      the earlier-diagnosed conflict where a hardcoded target of 1
#      actively fights channels whose true variance is far from 1.
#      --noise-step-start 0.05 --noise-step-end 0.01: Gaussian noise
#      injected at EVERY autoregressive step during Phase-2 training (not
#      just the rollout's starting pair) -- forces the model to
#      repeatedly recover from off-manifold perturbations, which a true
#      fixed point cannot do without real restoring dynamics. Never used
#      in this lineage before.
#
# Verified via a real smoke run before launching (2026-09-09): Phase 1 at
# w_shape_floor=0.3 and Phase 2 with --w-varmatch/--w-varmatch-adaptive/
# --noise-step-start/--noise-step-end all active train cleanly together
# (no NaN) at this exact K=24/N_w=64/L=22/degree=2/max_term_order=5
# sizing.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section113_spectralfield_L22_K24_Nw64_polynomial_anticollapse_euler

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=24, N_w=64) + REAL spectral_pde aux propagator (euler, ode_substeps=3, POLYNOMIAL field_kind, degree=2, max_order=4, max_term_order=5, --full-propagator), w_pred=1.5, w_shape_floor=0.3 (ESCALATED from 0.1), w_lowpass=0.0012, w_lowpass_rollout=0.02, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator euler --ode-substeps 3 \
  --spectral-field-kind polynomial --spectral-poly-degree 2 --spectral-poly-max-term-order 5 --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.3 \
  --w-lowpass-rollout 0.02 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Phase 2 (extend rollout): warm-started from Phase 1's own propagator, k_pred=2 -> k_max=12, w_lowpass_rollout=1.0, NEW anti-collapse terms (w_varmatch=0.1 adaptive, noise_step 0.05->0.01), --amp, 80 epochs, then Gate 3/4 ==="
STAGE2_TAG="${TAG}_extended"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp --k-max 12 --k-mid 8 --k-warmup-epochs 56 --k-mid-epochs 46 --epochs 80 \
  --w-lowpass-rollout 1.0 --lowpass-rollout-power 1.0 \
  --w-varmatch 0.1 --w-varmatch-adaptive \
  --noise-step-start 0.05 --noise-step-end 0.01 \
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

echo "=== Section 113 complete ==="
