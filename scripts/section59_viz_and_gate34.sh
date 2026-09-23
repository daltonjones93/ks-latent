#!/bin/zsh
# User-directed 2026-09-02: "please run visualizations and gate 3/4
# analysis on 59 now" -- Section 59 (Section 52's recipe with dt_snap=0.25,
# Stage 1 stopped at the epoch-99 mid-training checkpoint per user
# request rather than the full 200 epochs, then warm-started into Stage 2
# unchanged). Note k_max=12 covers only 12*0.25=3.0 physical time units
# here vs. 12.0 in every dt_snap=1.0 section, so val_kmax_mse/Gate 3/4
# numbers are not directly comparable to other sections without
# accounting for that -- see this script's TAG.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section59_mlpmarkovian_dtsnap025_wvar002_wspatialsigned01_logdet0035_200ep_epoch99ckpt_warmstart_k12_300ep
AE=artifacts/stage1_ae_patched_full_section59_mlpmarkovian_dtsnap025_wvar002_wspatialsigned01_logdet0035_200ep.pt
PROP=artifacts/stage2_prop_patched_full_${TAG}.pt
DATASET=artifacts/datasets/stage1_trajectories_dtsnap025.h5

echo "=== [1/2] Visualization: ${TAG} ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --dt-snap 0.25 \
  --tag "$TAG" > artifacts/logs/visualize_${TAG}.log 2>&1

echo "=== [2/2] Gate 3/4: ${TAG} ==="
mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$TAG" > artifacts/logs/gate3_analysis_${TAG}.log 2>&1 &
P1=$!
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$TAG" > artifacts/logs/gate3_da_${TAG}.log 2>&1 &
P2=$!
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$TAG" > artifacts/logs/gate4_diagnostics_${TAG}.log 2>&1 &
P3=$!
wait $P1 $P2 $P3

echo "=== Section 59 visualization + Gate 3/4 complete ==="
