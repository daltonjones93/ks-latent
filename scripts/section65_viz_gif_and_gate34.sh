#!/bin/zsh
# User-directed 2026-09-02/03: "run visualizations and gate 3/4
# diagnostics for 65" -- Section 65 (w_var=0.015, w_spatial_signed=0.035,
# w_logdet=0.005, lambda_z=0.001 active from EPOCH 0 (not delayed);
# cond#=1.212e7, D8=0.8261 (highest coherence in the sweep), but
# val_kmax_mse=0.088522 -- by far the worst Stage-2 fit in this family).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section65_mlpmarkovian_wvar0015_wspatialsigned0035_logdet0005_lambdaz0001fromepoch0_200ep
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

echo "=== Section 65 visualization + GIF + Gate 3/4 complete ==="
