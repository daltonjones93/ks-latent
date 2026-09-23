#!/bin/zsh
# User-directed 2026-09-05 (standing "visualizations, smoothness
# diagnostic + Gate 3/4" practice, applied to Section 93 given its
# result: best_val_kmax_mse=0.011800, best of the entire arc, AND the
# smoothest embedding of the entire project (stepN_med=0.1101,
# curvN_med=0.0555 -- beating even Section 92). Section 93 (= Section 92's
# architecture scaled down to ~55% params: d_model 116->84,
# fourier_hidden 185->130, same 75/25 ViT/Fourier ratio, token_mlp
# pooling both sides, Section 52's propagator unchanged). Same pattern as
# section92_viz_gate34.sh.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section93_vitfourieriffthybrid_dmodel84_fourierhidden130_tokenmlp_propmlpmarkovian52_wspatial019_wsmooth0005_200ep
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

echo "=== Section 93 visualization + GIF + Gate 3/4 (incl. D9) complete ==="
