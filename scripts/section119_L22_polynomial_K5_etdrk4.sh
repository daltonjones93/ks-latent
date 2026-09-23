#!/bin/zsh
# User-directed 2026-09-09, "Section 119": "should we try the etdrk4
# scheme with 117? maybe that would give more structure and non
# collapsing dynamics?" (after Section 118 -- Section 117's exact recipe
# plus poly_stable_leading -- fixed the NaN divergence but fell back into
# a full, stark collapse to a fixed point instead, same magnitude as
# Sections 114/116).
#
# Section 117's own recipe: K=5, d_latent=10, degree=3, max_term_order=5,
# poly_norm_power=0.5 (weakened normalization), w_logdet_physical=0.01 +
# w_logdet_physical_rollout=0.05 (Stage 1), w_var_physical=0.005 (Stage
# 1), w_logdet_rollout=0.01 (Stage 2) -- diverged to NaN during free
# rollout (worse with more training: step 22 at epoch 39, step 9 at
# epoch 79) under euler.
#
# This run swaps ONLY the integrator, euler -> etdrk4, keeping every
# other weight/architecture choice identical to Section 117 (no
# poly_stable_leading this round, to isolate etdrk4's own effect in
# contrast to Section 118's architectural fix).
#
# Why etdrk4 specifically addresses the divergence mechanism: under
# etdrk4, the LINEAR part of the dynamics is NOT parametrized by the
# learned polynomial's own linear coefficients at all -- it's handled by
# the EXACT, non-learned exp(dt*(k^2-k^4)) exponential propagator (KS's
# own true dispersion relation, `_build_etdrk4_coeffs`'s own Lhat,
# computed once via the same contour-integral formula the real solver
# uses), which is unconditionally numerically stable by construction. The
# learned polynomial correction only has to capture the NONLINEAR
# remainder, entering as a bounded forcing term into an already-stable
# exponential-integrator framework -- rather than under euler, where the
# SAME learned polynomial (including its own linear terms) gets applied
# via a naive explicit step with no protection against a badly-
# conditioned high-order coefficient. This is the SAME justification
# `_build_etdrk4_coeffs`'s own docstring gives for why etdrk4 is
# well-founded for a genuine spectral_field/spectral_pde pairing
# (unlike spectral_pde_raw's arbitrary self-FFT ordering, where etdrk4
# was found to diverge for the opposite reason -- an unjustified
# dispersion-relation assumption on a learned latent ordering).
#
# Verified via real smoke tests before launching: Phase 1 trains cleanly
# under etdrk4 (val_recon_final=0.156 after 3 epochs -- higher than
# euler's ~0.05-0.09 at this budget, not alarming, plausibly a different
# convergence rate); a 200-step rollout on that checkpoint completed
# WITHOUT diverging; Phase 2 continuation smoke test stayed finite
# (val_kmax_mse 0.011-0.016 across 5 epochs, growing with k rather than
# frozen).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section119_spectralfield_L22_K5_Nw64_polynomial_etdrk4

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=5, N_w=64, d_latent=10) + REAL spectral_pde aux propagator (ETDRK4 (up from euler), ode_substeps=3, POLYNOMIAL field_kind, degree=3, max_order=4, max_term_order=5, poly_norm_power=0.5, --full-propagator), w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, w_logdet_physical=0.01, w_logdet_physical_rollout=0.05, w_var_physical=0.005, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 5 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator etdrk4 --ode-substeps 3 \
  --spectral-field-kind polynomial --spectral-poly-degree 3 --spectral-poly-max-term-order 5 \
  --spectral-poly-norm-power 0.5 --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-logdet-physical 0.01 --w-logdet-physical-rollout 0.05 --w-var-physical 0.005 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
  --w-lowpass-rollout 0.02 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Phase 2 (extend rollout): warm-started from Phase 1's own propagator, k_pred=2 -> k_max=12, w_lowpass_rollout=1.0, w_logdet_rollout=0.01, --amp, 80 epochs, then Gate 3/4 ==="
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

echo "=== Section 119 complete ==="
