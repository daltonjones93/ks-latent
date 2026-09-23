#!/bin/zsh
# User-directed 2026-09-09. After Section 126's temporal_expansion_floor
# (real-data lag-1 separation floor, both z-space and w-space) still
# collapsed to a static pattern by epoch 39, the user asked a decisive
# clarifying question: "we could try a rollout variant, but it's the
# propagator already collapsing in stage 1? so it's sort of irrelevant?"
#
# This clarified what's actually worth testing: with `physics_prior`'s
# learned correction pinned near zero (correction_scale~0,
# --physics-prior-correction-warmup-epochs 100000, unchanged from Sections
# 124-126), the propagator's own step has NO free parameters left to
# adjust -- but the ENCODER still chooses WHERE in z-space real states
# land, and a fixed nonlinear map can be locally expansive in one region
# of phase space and contractive in another. So a "rollout variant" isn't
# irrelevant -- it (or, better, a LOCAL version of the same idea) tests
# whether the ENCODER is placing real states somewhere the EXISTING exact
# KS dynamics happens to expand rather than contract, which is a
# genuinely different question than anything tried in Sections 124-126
# (all of which only checked real-data properties like separation/
# covariance rank, never actually queried the DYNAMICS at all).
#
# NEW: `propagator_local_expansion_floor_loss`
# (ks_latent/training/losses.py) -- a DIFFERENTIABLE, one-sided floor on
# the propagator's own per-sample step-Jacobian spectral norm (largest
# singular value), evaluated at real encoded states, via
# torch.func.vmap(jacrev(...)) -- the SAME technique
# ks_latent.analysis.diagnostics.propagator_step_jacobian_spectral_norms
# already uses post-hoc for Gate 4's D9 diagnostic (prop_jacobian_med),
# now turned into a training-time regularizer. floor=1.0 is the absolute
# expand-vs-contract boundary (no real-data calibration needed, unlike
# the temporal floor's own reference).
#
# Verified directly (not just smoke-tested) before launching: a genuinely
# expanding toy map gets exactly zero loss; a genuinely contracting one
# gets penalized by the EXACT expected amount (relu(floor-scale)^2);
# gradient flows correctly back through a real spectral_pde propagator's
# own step_one into z. Found and fixed a real device-support gap along
# the way: torch.linalg.svdvals has no MPS kernel -- moved that one op to
# CPU (a differentiable device transfer, cheap given d~48 is small)
# rather than letting the whole run fail or forcing CPU-only training.
# EXPENSIVE (~0.5-1s per call measured directly, vs ~4.5-8s for a WHOLE
# epoch otherwise) -- applied only ONCE PER EPOCH, on a small (32-sample)
# subsample, not every batch; a real --amp smoke run confirmed negligible
# per-epoch overhead (4.6-5.3s vs ~4.5s baseline). 5 new unit tests pass;
# full suite re-run for regressions.
#
# Everything else identical to Sections 124-126: MLP inner encoder, K=24,
# N_w=64, etdrk4, physics_prior=True, correction_scale STILL pinned at ~0
# for the whole run (isolating whether this new mechanism ALONE, with
# zero learned dynamics, can keep the encoder from collapsing), w_pred at
# full weight from epoch 0, polynomial field_kind degree=2/max_term_order
# =5 (present but irrelevant while correction_scale~0), w_shape_floor=0.1,
# w_lowpass=0.0012/0.02. temporal_expansion_floor terms from Section 126
# DROPPED this round (isolating the new mechanism's own effect cleanly,
# matching this session's practice of changing one variable at a time).
#
# Phase-1-only (diagnostic), matching Sections 124-126's own pattern --
# check via visualize_rollout.py on early mid-training checkpoints before
# deciding whether to commit to a full Phase 2 + Gate 3/4 run.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section127_spectralfield_mlp_L22_K24_Nw64_physicsprior_localexpansionfloor_zerocorrection

echo "=== [1/1] Phase 1 (JOINT, DIAGNOSTIC): spectral_field AE with MLP inner model (K=24, N_w=64) + REAL spectral_pde aux propagator (ETDRK4, ode_substeps=3, PHYSICS_PRIOR ACTIVE, correction_scale PINNED AT ~0 for the whole run, w_pred at FULL weight from epoch 0), NEW w_local_expansion_floor=0.1 (floor=1.0, n_samples=32, differentiable Jacobian-spectral-norm floor via vmap(jacrev), applied once per epoch), polynomial correction degree=2 (irrelevant here), max_order=4, max_term_order=5, --full-propagator, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --spectral-field-inner mlp --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator etdrk4 --ode-substeps 3 \
  --spectral-physics-prior --physics-prior-correction-warmup-epochs 100000 \
  --spectral-field-kind polynomial --spectral-poly-degree 2 --spectral-poly-max-term-order 5 \
  --w-local-expansion-floor 0.1 \
  --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
  --w-lowpass-rollout 0.02 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

echo "=== Section 127 Phase 1 complete (diagnostic -- check mid-training checkpoints via visualize_rollout.py before deciding whether to continue to Phase 2) ==="
