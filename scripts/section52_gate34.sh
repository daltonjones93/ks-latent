#!/bin/zsh
# User-directed 2026-09-02: "run the visualization then gate 3/4" --
# Section 52 (w_spatial_signed=0.01, cond#=1.59e6, val_kmax_mse=0.025623
# -- first clear degradation in the spatial_signed sweep).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section52_mlpmarkovian_wvar002_wspatialsigned01_logdet0035_200ep.pt
TAG=section52_mlpmarkovian_wvar002_wspatialsigned01_logdet0035_200ep_warmstart_k12_300ep
PROP=artifacts/stage2_prop_patched_full_${TAG}.pt

echo "=== Gate 3/4: ${TAG} ==="
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

echo "=== Section 52 Gate 3/4 complete ==="
