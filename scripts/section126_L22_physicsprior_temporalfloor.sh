#!/bin/zsh
# User-directed 2026-09-09: "yeah build and test that, but design it for
# w space and for z space where w = irfft(z). I want to test both. kill
# 125 too."
#
# Context: Section 124's diagnostic (correction_scale pinned at ~0 for the
# WHOLE run -- propagator = literally the exact analytic KS equation, zero
# learned dynamics at all -- plus w_pred at full weight from epoch 0)
# still collapsed to a static pattern by epoch 19. Since there was no
# learned dynamics to blame, this conclusively implicated the ENCODER
# itself: under w_pred pressure, nothing in plain reconstruction loss
# stops it from mapping temporally-adjacent, causally-connected real
# states to nearly the same z, which trivially minimizes prediction error
# against ANY propagator (even a perfectly exact one). Section 125 added
# `w_logdet_physical` (a BATCH-level, cross-snapshot covariance-rank
# check) on top of this same setup -- STILL collapsed by epoch 29, because
# a batch mixing many unrelated snapshots can show full aggregate
# diversity while still flattening any SPECIFIC real trajectory segment --
# a different, more local granularity of collapse than a global covariance
# check can see.
#
# NEW: `temporal_expansion_floor_loss` (ks_latent/training/losses.py) --
# a direct, TEMPORAL-STRUCTURE-SPECIFIC fix. A ONE-SIDED floor (same
# design convention as spectral_shape_floor_loss/w_shape_floor -- real-
# data-calibrated, never penalizes exceeding it) on how close together, in
# representation space, real states `--temporal-floor-lag` (default 1)
# real steps apart are allowed to become. The reference floor
# (`reference_temporal_separation`) is computed ONCE from real ground-
# truth data via the SAME fixed transform used everywhere else (encode_
# to_spectrum/decode_from_spectrum), entirely independent of the current
# encoder -- so there's no circularity (the target doesn't move as the
# encoder trains).
#
# Built and tested BOTH scopings the user asked for, as separate,
# independently-weighted terms active simultaneously in this run:
#   - --w-temporal-floor-z 0.01: applies the floor in z-space (the rFFT
#     coefficients directly).
#   - --w-temporal-floor-w 0.01: applies the IDENTICAL mechanism in
#     w-space (decode_from_spectrum(z), the physical field) -- the same
#     z-vs-physical scoping distinction this project draws elsewhere
#     (w_logdet vs w_logdet_physical, w_var vs w_var_physical).
#
# Verified directly (2026-09-09, not just smoke-tested) before launching:
# a synthetic test confirmed the mechanism behaves exactly as intended --
# collapsed (constant-over-time) states get heavily penalized (loss~1.93),
# states matching the real reference distribution get a small residual
# loss (~0.076), and states MORE separated than the reference get EXACTLY
# zero loss (confirming the one-sided floor never fights legitimate
# expansion). 8 new unit tests (tests/unit/test_temporal_expansion_floor_
# loss.py) cover construction, gradient flow, one-sidedness, and lag>1;
# all pass. A real smoke run (5 epochs, both terms at weight 0.01) trained
# cleanly (val_recon_final=0.051, no NaN); a 200-step rollout on that
# checkpoint completed without diverging.
#
# Everything else identical to Sections 124/125: MLP inner encoder, K=24,
# N_w=64, etdrk4, physics_prior=True, correction_scale STILL pinned at ~0
# for the whole run (warmup=100000 epochs -- isolating whether this new
# regularizer alone, with zero learned dynamics, can keep the ENCODER's
# own representation from collapsing), w_pred at full weight from epoch 0
# (no warmup), polynomial field_kind degree=2/max_term_order=5 (present
# but irrelevant while correction_scale~0), w_shape_floor=0.1, w_lowpass=
# 0.0012/0.02.
#
# Phase-1-only (diagnostic), matching Sections 124/125's own pattern --
# check via visualize_rollout.py on early mid-training checkpoints (epoch
# ~19-39, where every prior collapse in this arc was already clearly
# visible) before deciding whether to commit to a full Phase 2 + Gate 3/4
# run.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section126_spectralfield_mlp_L22_K24_Nw64_physicsprior_temporalfloor_zerocorrection

echo "=== [1/1] Phase 1 (JOINT, DIAGNOSTIC): spectral_field AE with MLP inner model (K=24, N_w=64) + REAL spectral_pde aux propagator (ETDRK4, ode_substeps=3, PHYSICS_PRIOR ACTIVE, correction_scale PINNED AT ~0 for the whole run, w_pred at FULL weight from epoch 0), NEW w_temporal_floor_z=0.01 AND w_temporal_floor_w=0.01 (lag=1, real-data-calibrated one-sided floors on temporal separation, targeting the just-diagnosed encoder-representational-collapse mechanism directly), polynomial correction degree=2 (irrelevant here), max_order=4, max_term_order=5, --full-propagator, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --spectral-field-inner mlp --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator etdrk4 --ode-substeps 3 \
  --spectral-physics-prior --physics-prior-correction-warmup-epochs 100000 \
  --spectral-field-kind polynomial --spectral-poly-degree 2 --spectral-poly-max-term-order 5 \
  --w-temporal-floor-z 0.01 --w-temporal-floor-w 0.01 \
  --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
  --w-lowpass-rollout 0.02 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

echo "=== Section 126 Phase 1 complete (diagnostic -- check mid-training checkpoints via visualize_rollout.py before deciding whether to continue to Phase 2) ==="
