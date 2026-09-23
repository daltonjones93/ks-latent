#!/bin/zsh
# User-directed 2026-09-05 (standing "visualizations, smoothness
# diagnostic + Gate 3/4" practice, applied to Section 92 given its
# result: best_val_kmax_mse=0.013474, best of the entire 71-92 arc, AND
# the smoothest embedding of the entire project measured this session
# (stepN_med=0.1139, curvN_med=0.0580 -- beating even Section 52).
# Section 92 (= Section 91 + token_mlp pooling on both encoder and
# decoder). Same pattern as section91_viz_gate34.sh.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section92_vitfourieriffthybrid_dmodel116_fourierhidden185_tokenmlp_propmlpmarkovian52_wspatial019_wsmooth0005_200ep
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

echo "=== Section 92 visualization + GIF + Gate 3/4 (incl. D9) complete ==="
