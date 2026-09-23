#!/bin/zsh
# User-directed 2026-09-09, "Section 115": Section 114 (w_logdet_physical
# applied ONLY to the Stage-1 encoder's own real z) still showed the
# collapse signature -- val_kmax_mse frozen at ~0.0053 from Phase-2 epoch
# 26 through 79 (final), essentially flat despite k growing 6->12, the
# same "not growing with rollout horizon" signature every other collapsed
# run in this arc (106/112/113) showed before Gate 3 confirmed D_KY=0.
# (Section 114's own Gate 3 result was still computing when this was
# written -- check `docs/research-summary-sep-19.md` §6.6/6.7 or
# artifacts/logs/gate3_analysis_section114_..._extended.log for the final
# number.)
#
# User's diagnosis, correctly identifying the gap flagged as a caveat when
# 114 was launched: "the way I see it w_logdet_physical should apply to
# the encoder in stage 1, and the rollout from the propagator in stage 1,
# and also to the rollout in stage 2. for stage 2 we want the physical
# version of logdet." Section 114 only did the first of these three.
#
# This run applies the SAME logdet_barrier_loss(decode_from_spectrum(...))
# mechanism in all three places:
#   1. Stage 1, encoder's own z: --w-logdet-physical 0.01 (unchanged from
#      Section 114).
#   2. Stage 1, joint aux propagator's own short rollout z_pred: NEW
#      --w-logdet-physical-rollout 0.01 (Stage1TrainingConfig.
#      w_logdet_physical_rollout, added today -- mirrors the existing
#      w_lowpass/w_lowpass_rollout pairing).
#   3. Stage 2, the real propagator's own extended rollout z_pred: NEW
#      --w-logdet-rollout 0.01 (Stage2TrainingConfig.w_logdet_rollout,
#      built during the same conversation as Section 114 but not enabled
#      in that run -- this is the "physical version" the user asked for,
#      confirmed already implemented as decode_from_spectrum(z_pred) ->
#      logdet_barrier_loss, not the raw-z version).
#
# Verified via a real smoke run (2026-09-09) before launching: Phase 1
# with BOTH w_logdet_physical and w_logdet_physical_rollout active trains
# cleanly (no NaN); Phase 2 continuation with w_logdet_rollout ALSO active
# stayed in a healthy val_kmax_mse 0.028-0.040 range across 5 epochs --
# not the frozen-near-zero signature Section 114 showed at this point.
#
# Otherwise identical to Section 112/114's recipe (degree=2,
# max_term_order=5, normalized, w_shape_floor=0.1, no w_varmatch/
# noise_step -- already shown ineffective in Section 113, no delta_cap --
# already shown unnecessary, docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md
# Sections 15-16).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section115_spectralfield_L22_K24_Nw64_polynomial_wlogdetphysical_full_euler

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=24, N_w=64) + REAL spectral_pde aux propagator (euler, ode_substeps=3, POLYNOMIAL field_kind, degree=2, max_order=4, max_term_order=5, --full-propagator), w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, w_logdet_physical=0.01 (encoder z), NEW w_logdet_physical_rollout=0.01 (Stage-1 propagator rollout, physical space), dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator euler --ode-substeps 3 \
  --spectral-field-kind polynomial --spectral-poly-degree 2 --spectral-poly-max-term-order 5 --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-logdet-physical 0.01 --w-logdet-physical-rollout 0.01 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
  --w-lowpass-rollout 0.02 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Phase 2 (extend rollout): warm-started from Phase 1's own propagator, k_pred=2 -> k_max=12, w_lowpass_rollout=1.0, NEW w_logdet_rollout=0.01 (physical-space anti-collapse barrier on decode_from_spectrum(z_pred)), --amp, 80 epochs, then Gate 3/4 ==="
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

echo "=== Section 115 complete ==="
