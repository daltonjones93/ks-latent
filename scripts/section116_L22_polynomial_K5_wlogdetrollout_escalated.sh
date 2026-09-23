#!/bin/zsh
# User-directed 2026-09-09, "Section 116". Section 115 (K=24, degree=2,
# max_term_order=5, w_logdet_physical=0.01 + NEW w_logdet_physical_rollout
# =0.01 + Stage-2 w_logdet_rollout=0.01) showed, at a mid-Phase-1 check
# (epoch ~119/200), something genuinely different from every prior
# collapsed run in this arc: instead of freezing to a flat constant
# (Section 114's own signature), the rollout settles into a persistent,
# high-frequency, spatially fine-grained oscillation that keeps moving for
# the whole 200-step horizon -- still unphysical (doesn't track the true
# smooth, large-scale traveling-wave structure), but NOT a literal fixed
# point. User's read: "so w_logdet_physical_rollout is doing something."
#
# Hypothesis for why it's spurious high-frequency noise rather than real
# dynamics: at K=24 (d_latent=48), the model has far more spectral
# capacity than the true L=22 attractor uses (only wavenumbers m=1,2,3
# fall inside the actual KS instability band at this L -- see
# docs/research-summary-sep-9.md §6.3's benchmark-correction discussion).
# The log-det barrier only demands the ROLLOUT's covariance stay full
# rank/non-degenerate across the batch -- it has no preference for WHICH
# modes carry that variance, so the propagator can satisfy it cheaply by
# exciting high-index (physically meaningless) modes instead of the true
# low-order dynamics.
#
# User-directed fix, two changes together:
#   1. --spectral-K 5 (down from 24, d_latent=10 down from 48) -- removes
#      the physically-unjustified high-wavenumber modes from the
#      representation ENTIRELY, architecturally, rather than relying on a
#      loss term to suppress them after the fact. Still a small margin
#      above the ~3 truly unstable modes (per the benchmark-correction
#      discussion above).
#   2. --w-logdet-physical-rollout 0.05 (up from 0.01, 5x) -- escalating
#      the one term already shown to be doing SOMETHING, now on a much
#      smaller, more physically-constrained mode set where satisfying it
#      via spurious high-frequency noise is no longer architecturally
#      available.
#
# --w-logdet-physical (Stage 1 encoder) and Stage 2's --w-logdet-rollout
# left at their Section 115 values (0.01 each) -- only the two changes
# above are new here, isolating their combined effect against 115's own
# result once both finish.
#
# Verified via a real smoke run (2026-09-09) before launching: Phase 1
# trains cleanly at K=5 (no NaN, val_recon_final=0.064 after 3 epochs --
# higher than the K=24 baseline's ~0.024, expected given far less spectral
# capacity to reconstruct with); Phase 2 continuation with the escalated
# w_logdet_physical_rollout inheritance + w_logdet_rollout=0.01 stayed
# finite and moving (val_kmax_mse 0.011->0.021 across 5 epochs, growing
# with k rather than frozen).
#
# N_w kept at 64 (unchanged) -- this is the intermediate physical field's
# own resolution (independent of K, the number of spectral modes kept),
# no reason to shrink it just because K did.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section116_spectralfield_L22_K5_Nw64_polynomial_wlogdetrollout_escalated_euler

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=5, N_w=64, d_latent=10) + REAL spectral_pde aux propagator (euler, ode_substeps=3, POLYNOMIAL field_kind, degree=2, max_order=4, max_term_order=5, --full-propagator), w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, w_logdet_physical=0.01, w_logdet_physical_rollout=0.05 (ESCALATED 5x from Section 115's 0.01), dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 5 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator euler --ode-substeps 3 \
  --spectral-field-kind polynomial --spectral-poly-degree 2 --spectral-poly-max-term-order 5 --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-logdet-physical 0.01 --w-logdet-physical-rollout 0.05 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
  --w-lowpass-rollout 0.02 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Phase 2 (extend rollout): warm-started from Phase 1's own propagator, k_pred=2 -> k_max=12, w_lowpass_rollout=1.0, w_logdet_rollout=0.01 (unchanged from Section 115), --amp, 80 epochs, then Gate 3/4 ==="
STAGE2_TAG="${TAG}_extended"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp --k-max 12 --k-mid 8 --k-warmup-epochs 56 --k-mid-epochs 46 --epochs 80 \
  --w-lowpass-rollout 1.0 --lowpass-rollout-power 1.0 \
  --w-logdet-rollout 0.01 \
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

echo "=== Section 116 complete ==="
