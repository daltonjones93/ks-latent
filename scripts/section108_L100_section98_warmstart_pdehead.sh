#!/bin/zsh
# User-directed 2026-09-08, "Section 108" (after Section 107 collapsed
# TWICE -- Rev 1/2 at w_pde_distill=0.8/2.0 mutual, both confirmed
# collapsed via visualize_rollout.py's Hovmoller plots AND Gate 3's
# lyapunov_two_step_D_KY=0.0/n_positive=0/lambda1=-0.0047; Rev 3 at
# w_pde_distill=0.3 detached, Section 98's own regularizers restored,
# also looked collapse-suspicious on the same recon/loss-near-zero
# pattern before being killed): "seems like we're just collapsing then.
# can you make 108 use the same encoder, decoder and propagator as 98, I
# want L = 100, I want w_pde_distill = 0.1, detached. please use the same
# regularizer parameters as 98. you can kill any currently running
# trainings."
#
# Design: this time, LITERALLY reuse Section 98's own TRAINED weights
# (--init-ae-checkpoint/--init-aux-checkpoint), not just its architecture
# recipe -- Section 98 was itself trained at L=100 (stage1_trajectories_
# dtsnap1.h5, the canonical dataset), so for the first time in this
# investigation there's no domain mismatch to reconcile: warm-start
# directly onto the SAME data it was originally trained on. This is the
# lowest-risk configuration attempted so far -- Section 98's own
# encoder+propagator are a KNOWN-GOOD, already-validated starting point
# (this whole recovery effort has been trying to get back to something
# like it), and w_pde_distill=0.1 (down from Rev 3's 0.3, itself already
# down from Rev 1/2's 0.8/2.0) plus DETACHED (matching Rev 3, not Rev
# 1/2's mutual) is the gentlest version of this idea tried yet.
#
# Regularizers: Section 98's own exact values (w_decorr=0, w_var=0.02,
# w_spatial=0.04 --spatial-signed, w_var_floor=0, w_logdet=0.008,
# w_smooth=0.003) -- see scripts/section98_no_fourier_branch.sh, unchanged
# from that script's own recipe.
#
# Phase 1 is a FINE-TUNE (not fresh init) -- --init-ae-checkpoint/
# --init-aux-checkpoint load Section 98's own Phase-1 (--full-propagator
# sized) checkpoints; --pde-distill builds a fresh pde_head (no warm-start
# option exists for it yet, it always starts fresh/zero_init regardless).
# Given this starts from an already-converged model, far fewer epochs are
# needed than a from-scratch run -- 50 epochs (vs. Section 98's own 200)
# to let the new pde_distill term have a real but gentle effect without
# extensive retraining/drift risk.
#
# Verified via a real smoke run before launching (2026-09-08): warm-start
# combined with --pde-distill loads correctly (confirmed via the printed
# "loaded AE from.../loaded aux propagator from..." messages), trains
# cleanly, no NaN.
#
# Phase 2/3 schedule: IDENTICAL to every other section this session
# (Sections 95/98/107's own Phase 2: --epochs 300 --k-max 12
# --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175). Phase 3
# (docs/sine_transform_pde_plan.md §23): freeze-propagator refinement,
# both options at weight 1.0.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points.h5
AE98=artifacts/stage1_ae_patched_full_section98_vitonly_dmodel56_tokenmlp_propmlpmarkovian52_wspatial04_wsmooth003_200ep.pt
AUX98=artifacts/stage1_prop_full_section98_vitonly_dmodel56_tokenmlp_propmlpmarkovian52_wspatial04_wsmooth003_200ep.pt
TAG=section108_vit_L100_section98warmstart_pdehead_detached

echo "=== [1/4] Phase 1 (FINE-TUNE from Section 98's own trained weights, DETACHED pde_head): plain vit (d_model=56, d_latent=44) + mlp/markovian aux (full-propagator), Section 98's OWN regularizers, w_pde_distill=0.1 (detached), L=100, --amp, 50 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder vit --aux-backbone mlp --mode markovian \
  --d-model 56 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --init-ae-checkpoint "$AE98" --init-aux-checkpoint "$AUX98" \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.04 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.003 \
  --pde-distill --w-pde-distill 0.1 --pde-hidden 128 --pde-n-blocks 3 \
  --full-propagator --amp \
  --epochs 50 --checkpoint-every 10 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/4] Phase 2 (extend rollout ONLY -- pde_head not involved, matching Sections 95/98's own untouched Phase 2 recipe): k_max=12, 300 epochs, --amp ==="
STAGE2_TAG="${TAG}_extended"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [3/4] Phase 3 (docs/sine_transform_pde_plan.md §23), REVISED after watching the first attempt: --w-pde-distill 1.0 --w-pde-rollout 1.0 --k-warmup-epochs 1 (immediate k=2->12 jump) showed near-flat convergence (pde_distill 0.046->0.0457, pde_rollout 0.704->0.669 over 18 epochs after an initial 6x jump at the k=2->12 transition). Two fixes: (1) real k-ramp (--k-warmup-epochs 50, was 1) so pde_rollout's own autoregressive-chain difficulty increases gradually, mirroring why aux's own Phase 2 ramps; (2) w_pde_rollout cut 1.0->0.2 (pde_rollout's raw magnitude was ~15x pde_distill's at equal weight, likely dominating the combined gradient and starving pde_distill of dedicated signal) -- w_pde_distill stays 1.0 ==="
PHASE3_TAG="${TAG}_phase3refine"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$PROP" \
  --init-pdehead-checkpoint "$PDEHEAD" --freeze-propagator \
  --w-pde-distill 1.0 --w-pde-rollout 0.2 \
  --amp --epochs 100 --k-max 12 --k-warmup-epochs 50 \
  --tag "$PHASE3_TAG" \
  > artifacts/logs/stage2_${PHASE3_TAG}.log 2>&1

echo "=== [4/4] Gate 3/4 (on Phase 2's OWN propagator) ==="
mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --dt-snap 1.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1 &
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dt-snap 1.0 --L 100.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1 &
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --L 100.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1 &
wait

echo "=== Section 108 complete ==="
