#!/bin/zsh
# Section 48's result is the new best in the whole investigation
# (val_kmax_mse=0.016823, cond#=3.50) -- full Gate 3/4 to characterize it.
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section48_mlpmarkovian_wvar002_wspatialsigned0002_logdet0005_200ep.pt
TAG=section48_mlpmarkovian_wvar002_wspatialsigned0002_logdet0005_200ep_warmstart_k12_300ep
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

echo "=== Section 48 Gate 3/4 complete ==="
