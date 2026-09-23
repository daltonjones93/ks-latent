#!/bin/zsh
# User-directed 2026-09-09, "Section 114" (REVISED -- an earlier launch
# under this same filename mistakenly applied a Stage-2 rollout variant;
# user corrected: "wait, I wanted you to apply w_logdet to stage 1 to the
# irfft of z (call this w)". That earlier run was killed and its
# artifacts removed before completing any epochs.
#
# Sections 112 (bare) and 113 (w_shape_floor=0.3 + w_varmatch/noise_step)
# BOTH collapsed with the polynomial field_kind (degree=2, max_term_order
# =5, normalized) -- Gate 3: D_KY=0.0, lambda1=-0.40/-0.32. `delta_cap`
# was considered next but ruled out: `docs/PHASE2_ARCHITECTURE_EXPERIMENTS
# .md` Sections 15-16 already ran this exact ablation on a different
# backbone and found delta_cap is "not required anywhere in the pipeline
# to recover chaotic, accurate, DA-skillful latent dynamics" -- removing
# it gave the BEST result of that whole investigation.
#
# This run instead adds `Stage1TrainingConfig.w_logdet_physical` (new,
# 2026-09-09): the SAME `logdet_barrier_loss` mechanism as the existing
# `w_logdet` (already in every recipe here, currently OFF), but applied to
# `decode_from_spectrum(z, K, N_w)` -- the exact irfft reconstruction of
# the PHYSICAL field `w` -- instead of `z` (the rFFT coefficients)
# directly. User: "this was a big thing we found that prevented collapse
# before so we may as well try it" (referring to `w_logdet`'s own
# earlier-project history preventing latent channel collapse from
# fine-tuning, see memory).
#
# Worked out with the user first whether z-space vs. w-space logdet
# should even differ mathematically: if `decode_from_spectrum` were a
# SQUARE orthonormal map, `logdet(Cov(w)+eps*I)` would be EXACTLY equal to
# `logdet(Cov(z)+eps*I)` (identical loss, pointless to add). It's not
# square here (z: 2*K=48-dim, w: N_w=64-dim, K=24 < N_w//2+1=33, a genuine
# truncation) -- so Cov(w) is rank <=48 in the ambient 64-dim space (the
# extra `eps*I` eigenvalues in the unreachable directions are a fixed,
# z-independent additive constant, not real gradient signal), and rFFT/
# irFFT's non-uniform Parseval weighting (DC/Nyquist weight 1, other modes
# weight 2, plus 1/N_w scaling) makes the "live" part of w-space logdet a
# REWEIGHTED version of z-space logdet, not a rotated copy. Net: a real
# but MILD difference (per-mode reweighting), not a fundamentally
# different mechanism -- calibrated the weight accordingly (see below)
# rather than treating this as a wholly new untested regularizer class.
#
# ALSO FLAGGED (not yet resolved, worth watching in the results): this is
# a STAGE-1-ONLY (encoder-side) regularizer, shaping what the ENCODER
# does to real snapshots. The observed collapse is measured on the
# PROPAGATOR's free rollout in Phase 2, where the encoder is frozen -- so
# this has no DIRECT hold on the propagator's own autonomous dynamics,
# only an indirect one (a better-conditioned target manifold may be
# harder for the propagator to degenerate away from). Launched anyway
# per direct user instruction, as a cheap, already-smoke-tested try.
#
# Weight calibration: first smoke-tested at w_logdet_physical=0.1 (an
# arbitrary first guess) -- Phase 1's val_recon_final=0.168 after 3
# epochs, vs. ~0.024-0.037 for every prior smoke test at this same epoch
# budget, a real reconstruction hit. Given the near-equivalence argument
# above, recalibrated to the SAME range this project's own prior z-space
# w_logdet settled at (0.0035-0.02, Sections 44-90) rather than treating
# 0.1 as a reasonable default for a "new" mechanism: re-smoke-tested at
# w_logdet_physical=0.01 -- val_recon_final=0.023738 (back to normal), and
# a Phase-2 continuation smoke (5 epochs, bare, no anti-collapse terms)
# stayed in a healthy val_kmax_mse 0.05-0.09 range throughout (no early
# collapse signature, unlike 112/113's own smoke tests).
#
# Otherwise IDENTICAL to Section 112's bare recipe (no w_varmatch, no
# noise_step, no delta_cap, no Stage-2 w_logdet_rollout) -- isolating
# w_logdet_physical's own effect cleanly.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section114_spectralfield_L22_K24_Nw64_polynomial_wlogdetphysical_euler

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=24, N_w=64) + REAL spectral_pde aux propagator (euler, ode_substeps=3, POLYNOMIAL field_kind, degree=2, max_order=4, max_term_order=5, --full-propagator), w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, NEW w_logdet_physical=0.01 (logdet_barrier_loss on decode_from_spectrum(z), the encoder's own irfft-reconstructed physical field), dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator euler --ode-substeps 3 \
  --spectral-field-kind polynomial --spectral-poly-degree 2 --spectral-poly-max-term-order 5 --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-logdet-physical 0.01 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
  --w-lowpass-rollout 0.02 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Phase 2 (extend rollout): warm-started from Phase 1's own propagator, k_pred=2 -> k_max=12, w_lowpass_rollout=1.0, NO anti-collapse terms here (isolating w_logdet_physical's own effect from Phase 1), --amp, 80 epochs, then Gate 3/4 ==="
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

echo "=== Section 114 complete ==="
