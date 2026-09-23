#!/bin/zsh
# User-directed 2026-09-04/05: "run visualizations and gate 3/4 analysis
# for 87" -- Section 87 (= Section 52 + w_spatial=0.019, w_logdet=0.004,
# w_smooth=0.002; a further push beyond Section 86). Stage 2 finished:
# best_val_kmax_mse=0.035139, worse than both 52 (0.0256) and 86 (0.0283),
# cond#=4.71e6 (worse still) -- continuing the "pushing w_spatial too far
# hurts" pattern from 76/77/82. Same pattern as
# section81_viz_gate34.sh/section82_viz_gate34.sh/section85_viz_gate34.sh/
# section86_viz_gate34.sh.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section87_mlpmarkovian_wvar002_wspatial0019_wsmooth0002_logdet004_200ep
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

echo "=== Section 87 visualization + GIF + Gate 3/4 complete ==="
