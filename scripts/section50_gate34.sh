#!/bin/zsh
# User-directed 2026-09-01: "run visualizations the gate 3/4" -- Section
# 50 (w_var=0.02, w_logdet=0.0035, w_spatial (signed)=0.0065,
# val_kmax_mse=0.017986, cond#=1.15e5 -- meaningfully more collapsed than
# 48/49 but Stage-2 fit held up almost as well).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section50_mlpmarkovian_wvar002_wspatialsigned0065_logdet0035_200ep.pt
TAG=section50_mlpmarkovian_wvar002_wspatialsigned0065_logdet0035_200ep_warmstart_k12_300ep
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

echo "=== Section 50 Gate 3/4 complete ==="
