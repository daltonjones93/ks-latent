#!/bin/zsh
# User-directed 2026-08-31: "can you run gate 3 and 4 diagnostics after
# you generate visualizations for this run" -- Section 40's AE
# (w_spatial=0.085, w_logdet=0.001; min_eig=1.59e-4, cond#=3.78e4,
# l_spatial=0.368) + Stage 2 (k_max=12, val_kmax_mse=0.062532).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_history3_fullprop_wvar005_tw16_wspatial085_logdet0001_120ep.pt
PROP=artifacts/stage2_prop_patched_full_history3_fullprop_wvar005_tw16_wspatial085_logdet0001_120ep_mlpmarkovian_k12.pt
TAG=history3_fullprop_wvar005_tw16_wspatial085_logdet0001_120ep_mlpmarkovian_k12

echo "=== [Gate 3/4]: ${TAG} ==="
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

echo "=== Gate 3/4 for ${TAG} complete ==="
