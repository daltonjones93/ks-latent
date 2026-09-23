#!/bin/zsh
# User-directed 2026-09-09, "Section 112" (REVISED -- the original degree=1
# launch was killed immediately: "ah crap, you're right, we needed a
# degree 2 polynomial. please kill the training and run that with the
# normalization fix. but really only let the combined degree of the terms
# be less than 5 (so w_xxx * w_xxx or w_xxx*w_xxxx would have 0
# coefficients since they have combined degree 6, 7 respectively.)").
#
# Section 104's own exact recipe (K=24, N_w=64, L=22, euler, ode_substeps
# =3, w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, joint Phase-1/
# Phase-2, same Phase-2 schedule), with field_kind switched to
# "polynomial", degree=2 (NOW correct -- degree=1 was purely linear and
# structurally could not represent KS's own nonlinear term -w*w_x at all),
# max_order=4 (default, unchanged), PLUS a new combined-derivative-order
# truncation: --spectral-poly-max-term-order 5 (added 2026-09-09, see
# `ks_latent.models.propagator._polynomial_term_indices`'s docstring) --
# excludes any monomial whose derivative orders SUM to >=5 from the
# library entirely (e.g. w_xxx*w_xxx [combined 6], w_xxx*w_xxxx [combined
# 7]), while KS's own true terms (w*w_x combined 1, w_xx alone combined 2,
# w_xxxx alone combined 4) all survive comfortably. Verified directly
# (2026-09-09): at max_order=4 (5 derivative channels), this cuts the
# unrestricted degree=2 library from 21 terms down to 15 (6 physically-
# unmotivated high-combined-order cross terms excluded).
#
# Uses the NEWLY FIXED normalization (same session, earlier today): each
# derivative order divided by a FIXED (2*pi*K/L)**n before the library is
# built -- without it, a real Stage-2 run blew up to NaN within 2 epochs
# (degree=2's raw output reached ~1.3e7 at nominal coefficient scale).
# Re-verified via a real smoke run (Phase 1 + Stage-2 continuation)
# immediately before THIS launch, now WITH the max_term_order=5 truncation
# too: stable, no NaN, at this exact K=24/N_w=64/L=22 sizing (Stage-2
# smoke loss 185->46 over 5 epochs, finite throughout -- larger than
# degree=1's own smoke numbers, consistent with a genuinely undertrained
# richer model, not an instability).
#
# --w-lowpass-rollout KEPT at the same calibrated weights used since
# Section 111 (0.02 Phase 1 / 1.0 Phase 2) -- targets spurious high-
# WAVENUMBER content in z itself, an orthogonal concern to field_kind,
# still the fix for Section 104's own original D_KY=22-inflation finding.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section112_spectralfield_L22_K24_Nw64_polynomial_deg2_maxterm5_euler

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=24, N_w=64) + REAL spectral_pde aux propagator (euler, ode_substeps=3, POLYNOMIAL field_kind, degree=2, max_order=4, max_term_order=5, --full-propagator), w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator euler --ode-substeps 3 \
  --spectral-field-kind polynomial --spectral-poly-degree 2 --spectral-poly-max-term-order 5 --mode markovian \
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

echo "=== Section 112 complete ==="
