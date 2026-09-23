#!/bin/zsh
# User-directed 2026-09-03: "please run visualizations and gate 3/4
# diagnostics for 75 once it finishes and ask for permission before
# launching 76" -- Section 75 (masked fourier_mlp encoder@attn_window=8 +
# Fourier-IFFT-readout everywhere, propagator/decoder@window=22,
# lambda_z=0.0002 -- direct test of the lambda_z hypothesis after
# Sections 71/72/74 plateaued at val_kmax_mse ~0.15-0.29 with lambda_z=0.002).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section75_enc8_prop22_dec22_fourierifftreadout_wvar01_wspatialsigned01_logdet008_lambdaz00002_200ep
AE=artifacts/stage1_ae_patched_full_${TAG}.pt
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [1/3] Visualization: ${STAGE2_TAG} ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/visualize_${STAGE2_TAG}.log 2>&1

echo "=== [2/3] Latent state evolution GIF: ${TAG} ==="
mamba run -n da_env python scripts/make_latent_gif.py \
  --ae-checkpoint "$AE" \
  --tag "$TAG" --fps 12 --style line > artifacts/logs/gif_${TAG}.log 2>&1

echo "=== [3/3] Gate 3/4: ${STAGE2_TAG} ==="
mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1 &
P1=$!
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1 &
P2=$!
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1 &
P3=$!
wait $P1 $P2 $P3

echo "=== Section 75 visualization + GIF + Gate 3/4 complete ==="
