#!/bin/zsh
# User-directed 2026-09-09: "kill 124, add w_logdet_physical back, use
# everything else from 124 and see if we can avoid collapse."
#
# Section 124 (diagnostic): correction_scale pinned at ~0 for the whole
# run (propagator = EXACTLY the true KS equation, nothing learned in the
# dynamics at all) + w_pred at full weight from epoch 0 -- STILL
# collapsed to a static pattern by epoch 19. Since there was zero learned
# correction to blame, this conclusively implicated the ENCODER itself:
# under w_pred's pressure, it's free to map temporally-diverse real
# states to similar/nearby z values (nothing in plain reconstruction loss
# prevents this), which trivially minimizes prediction error against a
# FIXED, exact propagator regardless of whether the real dynamics are
# chaotic -- the collapse is a representational phenomenon in the
# encoder, not something happening in the (here, non-existent) learned
# dynamics.
#
# This run adds back `--w-logdet-physical 0.01` (Stage1TrainingConfig.
# w_logdet_physical, already built earlier this session, off in every
# physics_prior run so far to isolate the mechanism cleanly): the
# full-covariance log-det anti-collapse barrier, applied to
# decode_from_spectrum(z) -- the encoder's own REAL, per-batch encoded
# data, nothing to do with any rollout -- computed once per real batch,
# directly penalizing the encoder for letting its own real-data covariance
# lose rank. This directly targets the mechanism just diagnosed (encoder
# representational collapse), rather than anything about the dynamics or
# the correction, which is why it's being tested on TOP of Section 124's
# already-diagnostic "zero learned correction, full w_pred" setup rather
# than on top of any of the previous (correction-focused) fixes.
#
# Everything else identical to Section 124: MLP inner encoder, K=24,
# N_w=64, etdrk4, physics_prior=True, correction_scale pinned at ~0
# (warmup=100000 epochs -- still no learned dynamics at all, isolating
# w_logdet_physical's effect on the ENCODER cleanly, matching this
# session's practice of changing one variable at a time), polynomial
# field_kind degree=2/max_term_order=5 (present but irrelevant, never
# contributes while correction_scale~0), w_shape_floor=0.1, w_lowpass=
# 0.0012/0.02, w_pred at full weight from epoch 0 (no warmup).
#
# Verified via a real smoke run (2026-09-09): trains cleanly
# (val_recon_final=0.021 after 5 epochs), a 200-step rollout on that
# checkpoint completed without diverging.
#
# Phase-1-only (diagnostic), matching Section 124's own pattern -- check
# via visualize_rollout.py on early mid-training checkpoints (epoch
# ~19-39, where Section 124's own collapse was already clearly visible)
# before deciding whether to commit to a full Phase 2 + Gate 3/4 run.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section125_spectralfield_mlp_L22_K24_Nw64_physicsprior_wlogdetphysical_zerocorrection

echo "=== [1/1] Phase 1 (JOINT, DIAGNOSTIC): spectral_field AE with MLP inner model (K=24, N_w=64) + REAL spectral_pde aux propagator (ETDRK4, ode_substeps=3, PHYSICS_PRIOR ACTIVE, correction_scale PINNED AT ~0 for the whole run, w_pred at FULL weight from epoch 0), NEW w_logdet_physical=0.01 (encoder-side anti-collapse barrier on REAL data, targeting the just-diagnosed encoder-representational-collapse mechanism), polynomial correction degree=2 (irrelevant here), max_order=4, max_term_order=5, --full-propagator, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --spectral-field-inner mlp --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator etdrk4 --ode-substeps 3 \
  --spectral-physics-prior --physics-prior-correction-warmup-epochs 100000 \
  --spectral-field-kind polynomial --spectral-poly-degree 2 --spectral-poly-max-term-order 5 \
  --w-logdet-physical 0.01 \
  --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
  --w-lowpass-rollout 0.02 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

echo "=== Section 125 Phase 1 complete (diagnostic -- check mid-training checkpoints via visualize_rollout.py before deciding whether to continue to Phase 2) ==="
