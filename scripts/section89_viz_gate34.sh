#!/bin/zsh
# User-directed 2026-09-05: "please run visualizations, smoothness
# diagnostic + Gate 3/4 (in the future add smoothness diagnostic to gate
# 4)." Section 89 (= Section 85 + encoder Fourier branch truncated to
# enc_out_modes=8, local ViT pooling pool_window=8, w_spatial=0.014,
# w_smooth=0.0009, --prop-attn-window 44). Gate 4 now includes D9
# (real-trajectory smoothness) automatically -- see
# ks_latent/analysis/diagnostics.py's smoothness_diagnostic, wired into
# scripts/run_diagnostics.py this same turn. Same pattern as
# section81_viz_gate34.sh/.../section85_viz_gate34.sh/section86_viz_gate34.sh.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section89_vitfourieriffthybrid_encoutmodes8_poollocalw8_wspatial014_wsmooth0009_200ep
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

echo "=== [3/3] Gate 3/4 (D9 smoothness now included in Gate 4): ${STAGE2_TAG} ==="
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

echo "=== Section 89 visualization + GIF + Gate 3/4 (incl. D9) complete ==="
