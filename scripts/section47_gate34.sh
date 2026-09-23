#!/bin/zsh
# User-directed 2026-09-01: "run gate 3 and gate 4 tests on it" -- Section
# 47 (spatial_signed w_spatial=0.08, catastrophic collapse: min_eig=2.09e-6,
# cond#=1.07e7, only 2 effective dimensions; val_kmax_mse=0.201603).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section47_mlpmarkovian_wvar004_wspatialsigned008_logdet0002_200ep.pt
TAG=section47_mlpmarkovian_wvar004_wspatialsigned008_logdet0002_200ep_warmstart_k12_300ep
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

echo "=== Section 47 Gate 3/4 complete ==="
