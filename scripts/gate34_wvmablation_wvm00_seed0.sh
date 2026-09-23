#!/bin/zsh
# User-directed 2026-08-31: "can you run gate 3 and 4 analysis on
# [stage2_multiseed_wvmablation-wvm00_seed0.log] as well?" -- the fine-tuned
# AE's w_varmatch=0 / k_max=8 / 68-epoch run (converged, val_kmax_mse=0.036852).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_history3_wspatial005_finetune_from_vitaux_200ep.pt
PROP=artifacts/stage2_prop_patched_full_multiseed_wvmablation-wvm00_seed0.pt
TAG=multiseed_wvmablation-wvm00_seed0

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
