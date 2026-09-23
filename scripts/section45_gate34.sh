#!/bin/zsh
# User-directed 2026-09-01: "generate the visualizations, then run the
# gate 3/4 tests" -- Section 45 (w_spatial=0.14, 240/500 epochs,
# val_kmax_mse=0.026753).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section45_mlphistory2_wvar003_wspatial014_logdet0005_nodeltacap_240ep.pt
TAG=section45_mlphistory2_wvar003_wspatial014_logdet0005_nodeltacap_240ep_warmstart_k12_500ep
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

echo "=== Section 45 Gate 3/4 complete ==="
