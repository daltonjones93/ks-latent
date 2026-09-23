#!/bin/zsh
# User-directed 2026-09-08, "Section 109" (after Section 108's warm-started
# fine-tune showed no clear collapse in Phase 1/2, but Phase 3's pde_head
# refinement plateaued unhelpfully in BOTH attempts -- original
# --w-pde-distill 1.0 --w-pde-rollout 1.0 --k-warmup-epochs 1, and the
# ramp+rebalance fix --w-pde-rollout 0.2 --k-warmup-epochs 50 -- both
# converged to essentially the same plateau, pde_distill~=0.043,
# pde_rollout~=0.67, just reached more slowly with the fix): "doesn't look
# like that fix worked either. Can we try to train everything from
# scratch? instead of fine tuning 98? set the pde param to .05 in stage 1.
# or half of what it was in the past run. call this section 109." Then,
# after presenting the plan for approval (per explicit instruction "don't
# run anything until you clear it with me"): "let's lower w_spatial to
# .005, other than that launch 109."
#
# Design: TRAIN FROM SCRATCH (no --init-ae-checkpoint/--init-aux-checkpoint,
# unlike Section 108's warm-start from Section 98's own trained weights) --
# Section 98's own architecture recipe (plain vit, d_model=56, d_latent=44
# default, mlp/markovian aux, --full-propagator), on L=100 (same domain as
# Section 98/108).
#
# Regularizers: Section 98's own values EXCEPT w_spatial, user-directed:
#   w_var:     0.02 (unchanged from Section 98)
#   w_spatial: 0.04 (Section 98) -> 0.005 (user-directed this round --
#              between Section 98's own proven value and Section 107 Rev
#              1/2's 0.001, which coincided with collapse there, though
#              that was ALSO combined with a much larger mutual pde
#              weight; 0.005 is a deliberate middle ground now that
#              w_pde_distill is far gentler (0.05, detached) than that
#              earlier attempt's 0.8-2.0 mutual)
#   w_logdet:  0.008 (unchanged from Section 98)
#   w_smooth:  0.003 (unchanged from Section 98)
#   w_decorr/w_var_floor: 0 (unchanged, off)
#
# w_pde_distill: 0.1 (Section 108) -> 0.05 (first launch, "half of what it
# was in the past run") -> 0.008 (user-directed after killing the 0.05
# launch: "w_pde_distill is still too high. set it to .008 and rerun").
# DETACHED throughout (matches Section 108, not Section 107's mutual
# experiment).
#
# Phase 1: 200 epochs (Section 98's own from-scratch schedule -- Section
# 108 used only 50 since it was fine-tuning already-converged weights;
# this run needs the full schedule since it's training from scratch).
#
# Phase 2: IDENTICAL to every prior section's own Phase 2 (extend rollout,
# k_max=12, NO pde_head involvement -- --epochs 300 --k-max 12
# --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175).
#
# Phase 3 DELIBERATELY NOT queued here -- both of Section 108's Phase 3
# attempts (original weights/no-ramp, and the ramp+rebalance fix)
# converged to essentially the same plateau (pde_distill~=0.043,
# pde_rollout~=0.67). Rather than assume the same recipe will behave
# differently just because Phase 1/2 changed, run Phase 1+2+Gate 3/4
# first, look at the results (does a FROM-SCRATCH encoder+propagator with
# gentler w_pde_distill=0.008 give pde_head an easier fit than the warm-
# started one did?), and decide Phase 3's design (or whether to run it at
# all) as a separate, explicit step.
#
# Verified via a real smoke run before launching (2026-09-08): fresh vit/
# d_latent=44 + mlp aux (full-propagator) + spectral_pde_raw pde_head
# (K=23/L=44), w_spatial=0.005, w_pde_distill=0.008 detached, trains
# cleanly on real L=100 data, no NaN.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points.h5
TAG=section109_vit_L100_fromscratch_pdehead_detached

echo "=== [1/3] Phase 1 (FROM SCRATCH, DETACHED pde_head): plain vit (d_model=56, d_latent=44) + mlp/markovian aux (full-propagator), Section 98's regularizers except w_spatial=0.005 (was 0.04), w_pde_distill=0.008 (detached), L=100, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder vit --aux-backbone mlp --mode markovian \
  --d-model 56 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.005 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.003 \
  --pde-distill --w-pde-distill 0.008 --pde-hidden 128 --pde-n-blocks 3 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/3] Phase 2 (extend rollout ONLY -- pde_head not involved, matching Sections 95/98/108's own untouched Phase 2 recipe): k_max=12, 300 epochs, --amp ==="
STAGE2_TAG="${TAG}_extended"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [3/3] Gate 3/4 (on Phase 2's OWN propagator). Phase 3 (pde_head refinement) DELIBERATELY NOT run here -- to be designed as a separate step after reviewing these results, given Section 108's Phase 3 plateaued unhelpfully twice already ==="
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

echo "=== Section 109 (Phase 1/2 + Gate 3/4) complete -- Phase 3 to be designed separately ==="
