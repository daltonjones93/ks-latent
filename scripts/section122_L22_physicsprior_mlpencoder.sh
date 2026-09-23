#!/bin/zsh
# User-directed 2026-09-09, "I think we've artificially constrained the
# encoder quite a bit. Why don't we let the encoder be a general mlp and
# see if that measurably changes things. the vit might not be the right
# model for this. kill 121 and try this."
#
# Context: Section 120 (physics_prior+etdrk4, no warmup) showed a
# genuinely chaotic, untrained baseline (verified directly at
# correction_scale=0) but visibly DAMPED that chaos within ~20 training
# epochs. Section 121 (added a correction_scale homotopy/continuation
# warmup, ramping the learned correction in slowly) still collapsed by
# epoch 39, EVEN THOUGH correction_scale was only ~0.26 there (74% exact
# physics, 26% learned correction) -- suggesting the encoder itself may be
# drifting away from being a genuinely faithful physical-field
# representation under joint training, which would undermine
# physics_prior's own exactness guarantee even at a mostly-physics mix
# (nothing besides plain reconstruction loss anchors the encoder's
# Fourier-coefficient semantics to what physics_prior assumes).
#
# NEW: `SpectralFieldAutoencoderConfig` generalized (was hard-coded to a
# ViT) to accept a plain, fully-connected `KSAutoencoderMLP` as the
# field-producing inner model instead (`--spectral-field-inner mlp`) --
# no attention, tokenization, or windowing at all, every output position
# can depend on every input position with no architectural locality bias
# the ViT's own windowed-attention/patch structure might be imposing.
# Both `KSAutoencoderViT` and `KSAutoencoderMLP` expose the identical
# encode/decode interface, so this is a clean drop-in swap -- isolates
# whether the ViT's OWN architecture (not physics_prior, not the warmup)
# was contributing to the collapse.
#
# Keeps Section 121's correction_scale warmup (150/200 epochs) -- testing
# the encoder swap ON TOP OF the already-implemented fix, not instead of
# it, since both address different (complementary) parts of the same
# diagnosed mechanism.
#
# Verified via real smoke tests before launching (2026-09-09): the MLP
# inner model trains noticeably FASTER per-epoch than the ViT (4.7s vs
# ~17-20s at this K=24/N_w=64 sizing) and converges cleanly
# (val_recon_final=0.043 after 3 epochs); a 200-step rollout on that
# checkpoint completed without diverging; Phase 2 continuation held
# val_kmax_mse~0.037-0.038 across 5 epochs, finite throughout. Full test
# suite (631 tests) passes, no regressions from the config/model changes
# (SpectralFieldAutoencoderConfig now requires exactly one of vit/mlp;
# 23 new/updated tests cover the mlp branch directly).
#
# Otherwise identical to Section 121: K=24, N_w=64, etdrk4,
# spectral_physics_prior=True, correction warmup over 150/200 epochs,
# polynomial correction degree=2, max_term_order=5, standard
# normalization, w_shape_floor=0.1, no other new regularizers.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section122_spectralfield_mlp_L22_K24_Nw64_physicsprior_correctionwarmup

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE with MLP inner model (K=24, N_w=64, NO ViT -- plain fully-connected encoder/decoder) + REAL spectral_pde aux propagator (ETDRK4, ode_substeps=3, PHYSICS_PRIOR ACTIVE, correction_scale warmup 0.0->1.0 over first 150/200 epochs, polynomial correction degree=2, max_order=4, max_term_order=5, --full-propagator), w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --spectral-field-inner mlp --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
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

echo "=== Section 122 complete ==="
