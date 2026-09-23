#!/bin/zsh
# User-directed 2026-09-01: "please run gate 3/4 tests on section 46
# result, after plotting visualizations for this model" -- Section 46
# (mlp/markovian aux, w_var=0.04, w_spatial=0.08, w_logdet=0.002,
# val_kmax_mse=0.027402, but a severely collapsed channel:
# min_eig=8.26e-6, cond#=7.5e5).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section46_mlpmarkovian_wvar004_wspatial008_logdet0002_200ep.pt
TAG=section46_mlpmarkovian_wvar004_wspatial008_logdet0002_200ep_warmstart_k12_300ep
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

echo "=== Section 46 Gate 3/4 complete ==="
