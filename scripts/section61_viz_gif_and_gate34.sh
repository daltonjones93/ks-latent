#!/bin/zsh
# User-directed 2026-09-02: "visualize 61, including the gif, run the
# gate 3/4 diagnostics" -- Section 61 (w_var decay 0.05->0.01, w_spatial
# (signed) decay 0.025->0.002, w_logdet decay 0.02->0.0005, 200ep Stage 1;
# best_val_kmax_mse=0.023340, back in the healthy range after Section 60's
# w_spatial decay-starting-at-the-catastrophic-0.08 mistake).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section61_mlpmarkovian_wvardecay005to001_wspatialsigneddecay0025to0002_logdetdecay002to00005_200ep
AE=artifacts/stage1_ae_patched_full_${TAG}.pt
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [1/3] Visualization: ${STAGE2_TAG} ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/visualize_${STAGE2_TAG}.log 2>&1

echo "=== [2/3] Latent state evolution GIF: ${TAG} ==="
mamba run -n da_env python scripts/make_latent_gif.py \
  --ae-checkpoint "$AE" \
  --tag "$TAG" --fps 12 --style line > artifacts/logs/gif_${TAG}.log 2>&1

echo "=== [3/3] Gate 3/4: ${STAGE2_TAG} ==="
mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1 &
P1=$!
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1 &
P2=$!
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1 &
P3=$!
wait $P1 $P2 $P3

echo "=== Section 61 visualization + GIF + Gate 3/4 complete ==="
