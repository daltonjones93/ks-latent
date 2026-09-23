#!/bin/zsh
# User-directed 2026-09-09, "Section 118". Section 117 (K=5, degree=3,
# poly_norm_power=0.5, w_var_physical=0.005 -- otherwise Section 116's
# recipe) did NOT collapse to a fixed point, but instead diverged to NaN
# during free rollout -- and the divergence point got WORSE with more
# training (step 22 at epoch 39, step 9 at epoch 79), confirming a real,
# worsening instability rather than something training resolves alone.
# Killed by the user before completing, after explicitly preferring this
# failure mode over collapse: "I'd rather see divergence and some kind of
# accurate modeling than just collapsing to a point. we can regularize the
# divergence easier than the collapse I think."
#
# User's follow-up: "is there a way to regularize or bound the eigenvalues
# of the differential operator induced by the pde?" -- YES: the
# polynomial's linear (degree-1) part is a genuine constant-coefficient
# differential operator, so each Fourier mode is an eigenfunction with
# eigenvalue lambda(k) = sum_n c_n*(ik)^n. Only EVEN n contributes to
# Re(lambda(k)) (odd orders are purely imaginary/dispersive). For
# boundedness as energy reaches high wavenumber, Re(lambda(k)) must go to
# -inf as k->inf, which reduces to: the coefficient on the HIGHEST kept
# even-order derivative (here, w_xxxx, order 4) must be negative -- KS's
# own true sign. A positive coefficient there is a direct, closed-form
# explanation for exactly the observed "trains fine short-horizon,
# diverges over a longer free rollout" pattern.
#
# NEW: --spectral-poly-stable-leading (Stage1TrainingConfig/
# AuxPropagatorConfig/PropagatorConfig.spectral_poly_stable_leading,
# ks_latent.models.propagator._SpectralPDEDeltaBody's poly_stable_leading)
# -- architecturally forces this one coefficient to `-(raw)^2` (always
# <=0, exactly 0 at zero_init, preserving the exact-identity-at-init
# invariant) regardless of what training does. Does NOT suppress
# instability at lower wavenumbers (still needed for genuine chaos) --
# targets only the runaway-blowup failure mode, not another collapse-
# inducing constraint. Verified: 27/27 polynomial unit tests pass
# (including new coverage for this feature), 225/225 propagator/config
# tests pass (no regressions), zero_init identity preserved directly.
#
# Verified via real smoke tests before launching: Phase 1 trains cleanly
# (val_recon_final=0.086 after 3 epochs); a 200-step rollout on that same
# 3-epoch checkpoint completed WITHOUT diverging (vs. Section 117's own
# checkpoint diverging by step 22-9 at this point in training); Phase 2
# continuation smoke test stayed finite (val_kmax_mse 0.006-0.012 across 5
# epochs).
#
# Otherwise identical to Section 117: K=5 (d_latent=10), degree=3,
# max_term_order=5, poly_norm_power=0.5, w_logdet_physical=0.01 +
# w_logdet_physical_rollout=0.05 (Stage 1), w_var_physical=0.005 (Stage
# 1), w_logdet_rollout=0.01 (Stage 2), w_shape_floor=0.1, w_lowpass=0.0012,
# w_lowpass_rollout=0.02/1.0.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section118_spectralfield_L22_K5_Nw64_polynomial_stableleading_euler

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=5, N_w=64, d_latent=10) + REAL spectral_pde aux propagator (euler, ode_substeps=3, POLYNOMIAL field_kind, degree=3, max_order=4, max_term_order=5, poly_norm_power=0.5, NEW poly_stable_leading=True (bounds w_xxxx coefficient <=0, prevents runaway high-wavenumber blowup), --full-propagator), w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, w_logdet_physical=0.01, w_logdet_physical_rollout=0.05, w_var_physical=0.005, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 5 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator euler --ode-substeps 3 \
  --spectral-field-kind polynomial --spectral-poly-degree 3 --spectral-poly-max-term-order 5 \
  --spectral-poly-norm-power 0.5 --spectral-poly-stable-leading --mode markovian \
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

echo "=== Section 118 complete ==="
