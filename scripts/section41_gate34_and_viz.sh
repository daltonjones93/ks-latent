#!/bin/zsh
# User-directed 2026-09-01: "once this training is done, please launch
# gate 3 and 4 diagnostics and visualize the results." Waits on the
# currently-running Section 41 pipeline (Stage 1 w_spatial=0.06/
# w_logdet=0.002/140ep -> Stage 2 mlp/markovian/k_max=12/200ep/no
# delta_cap), then runs Gate 3/4 + visualize_rollout.py on the result.
set -e
cd /Users/daltonjones/Documents/latent_DA

echo "=== waiting for the Section 41 training pipeline (PID 17281) to finish ==="
while kill -0 17281 2>/dev/null; do sleep 15; done

AE=artifacts/stage1_ae_patched_full_history3_fullprop_wvar005_tw16_wspatial06_logdet0002_140ep.pt
TAG=history3_fullprop_wvar005_tw16_wspatial06_logdet0002_140ep_mlpmarkovian_k12_200ep
PROP=artifacts/stage2_prop_patched_full_${TAG}.pt

echo "=== [1/2] Gate 3/4: ${TAG} ==="
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

echo "=== [2/2] Visualization: ${TAG} ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$TAG" > artifacts/logs/visualize_${TAG}.log 2>&1

echo "=== Section 41 Gate 3/4 + visualization complete ==="
