#!/bin/zsh
# Section 44's result is the best in the whole investigation
# (val_kmax_mse=0.022067) -- full Gate 3/4 + visualization to characterize
# it as the new standing best.
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section44_mlphistory2_wvar003_wspatial01_logdet0005_nodeltacap.pt
TAG=section44_mlphistory2_wvar003_wspatial01_logdet0005_nodeltacap_warmstart_k12_300ep
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

echo "=== Section 44 Gate 3/4 + visualization complete ==="
