#!/bin/zsh
# Diagnostic run, 2026-09-09 (not a "fix attempt" -- a decisive isolation
# test). Section 123 (MLP encoder, w_pred warmup 0->1.5 over 40 epochs,
# correction_scale warmup 0->1.0 over 150 epochs) already showed the
# familiar collapse-to-a-static-pattern signature by epoch 19, even
# though correction_scale was only ~0.13 there (mostly pure physics) and
# w_pred was already ~48% of its full weight. This raised a sharper
# question than either warmup alone can answer: is the collapse coming
# from the LEARNED CORRECTION (however small), or from the ENCODER ITSELF
# drifting away from a faithful physical-field representation once real
# w_pred pressure is applied -- independent of the correction entirely?
#
# This run answers that directly: `--physics-prior-correction-warmup-
# epochs 100000` pins `correction_scale` at ~0 for the ENTIRE run (a
# 200-epoch run reaches epoch/100000 ~= 0.002, indistinguishable from
# exactly 0) -- the propagator is EXACTLY the true KS equation throughout,
# nothing ever learned in the dynamics at all -- while w_pred is at its
# FULL weight from epoch 0 (no w_pred warmup this time), the worst case
# for encoder drift. If the rollout still loses its chaos under this
# setup, that conclusively implicates the encoder's own drift, not the
# learned correction, since there is no correction to blame at all.
#
# MLP inner encoder kept (already ruled out as the root cause, and
# trains ~4x faster, useful for getting a fast read here).
#
# Verified via a real smoke run (2026-09-09): trains cleanly
# (val_recon_final=0.021 after 5 epochs).
#
# Check via visualize_rollout.py on an EARLY mid-training checkpoint
# (epoch ~19-39) -- Section 123's own collapse was already visible by
# epoch 19, so this should give a fast, decisive read without needing
# the full 200-epoch schedule.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section124_spectralfield_mlp_L22_K24_Nw64_physicsprior_diagnostic_zerocorrection

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE with MLP inner model (K=24, N_w=64) + REAL spectral_pde aux propagator (ETDRK4, ode_substeps=3, PHYSICS_PRIOR ACTIVE, correction_scale PINNED AT ~0 for the whole run (warmup=100000 epochs), w_pred at FULL weight from epoch 0 (no warmup) -- isolating whether encoder drift alone, with ZERO learned correction ever, breaks the chaos), polynomial correction degree=2 (irrelevant here, never contributes), max_order=4, max_term_order=5, --full-propagator, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --spectral-field-inner mlp --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator etdrk4 --ode-substeps 3 \
  --spectral-physics-prior --physics-prior-correction-warmup-epochs 100000 \
  --spectral-field-kind polynomial --spectral-poly-degree 2 --spectral-poly-max-term-order 5 \
  --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
  --w-lowpass-rollout 0.02 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

echo "=== Section 124 Phase 1 complete (diagnostic -- check mid-training checkpoints via visualize_rollout.py before deciding whether to continue to Phase 2) ==="
