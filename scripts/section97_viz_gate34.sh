#!/bin/zsh
# User-directed 2026-09-05 (standing "visualizations + Gate 3/4"
# practice, applied to Section 97). Section 97 (= Section 96 scaled to
# 75% size, w_smooth=0.003 added back at 2x Section 95's value) landed
# close to Section 96 (val_kmax_mse=0.031086 vs 96's 0.029794, cond#=161
# vs 160) -- the size/w_smooth changes didn't meaningfully move the
# needle at this w_spatial=0.04 level. Same pattern as
# section96_viz_gate34.sh.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section97_vitfourieriffthybrid_dmodel56_fourierhidden70_tokenmlp_propmlpmarkovian52_wspatial04_wsmooth003_200ep
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

echo "=== [3/3] Gate 3/4 (incl. D9 smoothness): ${STAGE2_TAG} ==="
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

echo "=== Section 97 visualization + GIF + Gate 3/4 (incl. D9) complete ==="
