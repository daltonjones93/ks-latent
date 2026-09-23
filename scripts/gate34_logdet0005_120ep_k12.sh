#!/bin/zsh
# User-directed 2026-08-31: "how do the other gate 3 and 4 diagnostics
# look?" -- Section 39b's retuned AE (w_spatial=0.05, w_logdet=0.005,
# 120 epochs; cond#=62.4, D7 p=0.0000, l_spatial=0.411) + Stage 2
# (k_max=12, val_kmax_mse=0.030295, best result in the whole document).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_history3_fullprop_wvar005_tw16_wspatial005_logdet0005_120ep.pt
PROP=artifacts/stage2_prop_patched_full_history3_fullprop_wvar005_tw16_wspatial005_logdet0005_120ep_mlpmarkovian_k12.pt
TAG=history3_fullprop_wvar005_tw16_wspatial005_logdet0005_120ep_mlpmarkovian_k12

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
