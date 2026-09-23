#!/bin/zsh
# User-directed 2026-09-03: "can you run gate 3/4 and visualizations for
# this stage 2 run [...mlp_history5...] and this stage 2 run
# [...fnomlp_prop...]" -- both are propagator-architecture comparisons on
# Section 65's AE (mlp/history5: val_kmax_mse=0.068366, the best of the
# three backbones tried on this AE; fno_mlp: val_kmax_mse=0.257397,
# comparable to the earlier failed vit attempt at ~0.22).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section65_mlpmarkovian_wvar0015_wspatialsigned0035_logdet0005_lambdaz0001fromepoch0_200ep.pt

for TAG in section65_ae_mlp_history5_k12_300ep section65_ae_fnomlp_prop_k12_300ep; do
  PROP=artifacts/stage2_prop_patched_full_${TAG}.pt

  echo "=== [${TAG}] Visualization ==="
  mamba run -n da_env python scripts/visualize_rollout.py \
    --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
    --tag "$TAG" > artifacts/logs/visualize_${TAG}.log 2>&1

  echo "=== [${TAG}] Gate 3/4 ==="
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
  echo "=== [${TAG}] complete ==="
done

echo "=== Section 65 mlp/history5 + fno_mlp visualization + Gate 3/4 all complete ==="
