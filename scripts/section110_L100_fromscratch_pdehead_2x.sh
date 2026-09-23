#!/bin/zsh
# User-directed 2026-09-08, "Section 110": identical to Section 109
# (scripts/section109_L100_fromscratch_pdehead.sh -- from-scratch, plain
# vit/d_model=56/d_latent=44 + mlp/markovian aux, Section 98's regularizers
# except w_spatial=0.005, L=100), except Phase 1's w_pde_distill DOUBLED:
# 0.008 -> 0.016. Motivated directly by Section 109's own measured result:
# the propagator itself came out genuinely healthy (Gate 3/4 confirmed via
# Hovmoller + error-growth visualizations -- no collapse, val_kmax_mse=
# 0.0187, BETTER than Section 98's own 0.0332), but a direct measurement
# of pde_head's OWN fit quality against the final Phase-2 propagator
# showed it had learned almost nothing: MSE(pde_head vs aux) = 0.0725,
# essentially equal to aux's own real step-size MSE (0.0737, ratio 0.984)
# -- pde_head was behaving close to a no-op. User: "let's run 110 as the
# same as 109 but double the pde term in phase 1. launch this now."
#
# Verified via a real smoke run before launching (2026-09-08): fresh vit/
# d_latent=44 + mlp aux (full-propagator) + spectral_pde_raw pde_head
# (K=23/L=44), w_spatial=0.005, w_pde_distill=0.016 detached, trains
# cleanly on real L=100 data, no NaN.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points.h5
TAG=section110_vit_L100_fromscratch_pdehead_2x

echo "=== [1/3] Phase 1 (FROM SCRATCH, DETACHED pde_head, 2x pde weight): plain vit (d_model=56, d_latent=44) + mlp/markovian aux (full-propagator), Section 98's regularizers except w_spatial=0.005 (was 0.04), w_pde_distill=0.016 (detached, doubled from Section 109's 0.008), L=100, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder vit --aux-backbone mlp --mode markovian \
  --d-model 56 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.005 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.003 \
  --pde-distill --w-pde-distill 0.016 --pde-hidden 128 --pde-n-blocks 3 \
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

echo "=== Section 110 (Phase 1/2 + Gate 3/4) complete -- Phase 3 to be designed separately ==="
