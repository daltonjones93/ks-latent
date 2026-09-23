#!/bin/zsh
# User-directed 2026-09-09, "Section 117": "can you try 116 again with
# higher degree polynomial for the pde, and less normalization for the
# polynomial? also add a w_var term for the irfft of the latent state z."
#
# Section 116 (K=5, d_latent=10, degree=2, max_term_order=5, full
# normalization, w_logdet_physical=0.01 + w_logdet_physical_rollout=0.05
# Stage 1, w_logdet_rollout=0.01 Stage 2) collapsed to an extreme,
# saturated constant by mid-Phase-1 (epoch ~69/200), same signature as
# Section 114 -- killed by the user before completing.
#
# This run keeps Section 116's architecture/K/weights and adds three
# NEW capabilities built today:
#
#   1. --spectral-poly-degree 3 (up from 2) -- the polynomial library
#      construction (ks_latent.models.propagator._polynomial_term_indices/
#      _polynomial_library) was generalized from a hand-written degree-1/2
#      case split to a general itertools.combinations_with_replacement
#      construction supporting any degree; validated up to 3 (cubic cross
#      terms, e.g. w_x*w_xx*w_xxx). --spectral-poly-max-term-order 5 still
#      applies (combined order truncation), so most of the new cubic terms
#      this adds are still excluded -- only low-combined-order cubic terms
#      survive.
#   2. --spectral-poly-norm-power 0.5 (down from the implicit 1.0) --
#      generalizes the per-order normalization exponent from a fixed `n`
#      to `n*power`; 0.5 weakens it, letting higher-order derivative
#      channels keep more relative dynamic range. VERIFIED SAFE via a
#      direct scale test before launching (not just a smoke run, per this
#      mechanism's own docstring warning): at this K=5/L=22 sizing,
#      char_k=1.43 (vs K=24's own char_k=6.86, where the original blowup
#      was found) -- even fully UNnormalized (power=0.0) output stayed
#      ~1000 at a stress-tested coefficient/z scale, nowhere near the
#      original 1e7-1e12 blowup magnitude. power=0.5 is a verified-safe
#      middle ground at this K.
#   3. --w-var-physical (NEW, Stage1TrainingConfig.w_var_physical):
#      decorr_var_loss's per-channel variance-vs-1 term (--w-var's own
#      mechanism), applied to decode_from_spectrum(z) -- the encoder's own
#      physical-space field -- instead of z directly. Targets a DIFFERENT
#      collapse signature than w_logdet_physical (marginal per-point
#      variance floor vs. full-covariance rank).
#
# Weight calibration for w_var_physical, done via real smoke tests (not
# guessed): an initial 0.05 (same order of magnitude as the escalated
# w_logdet_physical_rollout) stalled Phase-1 reconstruction almost
# completely (val_recon_final=1.003 after 3 epochs, ~mean-baseline level,
# vs. ~0.06 for every other recent smoke test at this budget) -- isolated
# directly (a control run with everything else identical but
# w_var_physical=0 recovered normal convergence, confirming this term was
# the cause). Rescaled to 0.005 (10x down) restored normal convergence
# (val_recon_final=0.049); Phase-2 continuation smoke test stayed finite
# with val_kmax_mse in a small, reasonable 0.008-0.012 range across 5
# epochs.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section117_spectralfield_L22_K5_Nw64_polynomial_degree3_wvar_euler

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=5, N_w=64, d_latent=10) + REAL spectral_pde aux propagator (euler, ode_substeps=3, POLYNOMIAL field_kind, DEGREE=3 (up from 2), max_order=4, max_term_order=5, poly_norm_power=0.5 (LESS normalization, down from 1.0), --full-propagator), w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, w_logdet_physical=0.01, w_logdet_physical_rollout=0.05, NEW w_var_physical=0.005, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 5 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator euler --ode-substeps 3 \
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

echo "=== [2/3] Phase 2 (extend rollout): warm-started from Phase 1's own propagator, k_pred=2 -> k_max=12, w_lowpass_rollout=1.0, w_logdet_rollout=0.01 (unchanged from Section 115/116), --amp, 80 epochs, then Gate 3/4 ==="
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

echo "=== Section 117 complete ==="
