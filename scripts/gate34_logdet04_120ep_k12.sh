#!/bin/zsh
# User-directed 2026-08-31: "can we run the gate 3 and gate 4 on the log
# det phase 2 run" -- Section 38's log-det-barrier AE (min_eig=0.559,
# cond#=6.51) + fresh mlp/markovian Stage 2 propagator (k_max=12,
# val_kmax_mse=0.037618, matching the best result in the whole document).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_history3_fullprop_wvar005_tw16_wspatial005_logdet04_120ep.pt
PROP=artifacts/stage2_prop_patched_full_history3_fullprop_wvar005_tw16_wspatial005_logdet04_120ep_mlpmarkovian_k12.pt
TAG=history3_fullprop_wvar005_tw16_wspatial005_logdet04_120ep_mlpmarkovian_k12

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

echo "=== [Visualization]: ${TAG} ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$TAG" > artifacts/logs/visualize_${TAG}.log 2>&1

echo "=== Gate 3/4 + visualization for ${TAG} complete ==="
