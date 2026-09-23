#!/bin/zsh
# User-directed 2026-09-02: "please run visualizations then gate 3/4
# diagnostics when this run completes" -- Section 53 (bigger models:
# ViT AE d_model=108 + mlp/markovian aux hidden=141/n_blocks=3, both
# ~1.2x default param counts; w_var=0.025, w_logdet=0.0045,
# w_spatial_signed=0.01). Waits on the currently-running Section 53
# pipeline (PID 36468: Stage 1 -> spectrum/D7/D8 check -> Stage 2), then
# runs visualize_rollout.py followed by Gate 3/4.
set -e
cd /Users/daltonjones/Documents/latent_DA

echo "=== waiting for the Section 53 training pipeline (PID 36468) to finish ==="
while kill -0 36468 2>/dev/null; do sleep 15; done

TAG=section53_mlpmarkovian_bigger12x_dmodel108_wvar0025_wspatialsigned01_logdet0045_200ep
AE=artifacts/stage1_ae_patched_full_${TAG}.pt
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [1/2] Visualization: ${STAGE2_TAG} ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/visualize_${STAGE2_TAG}.log 2>&1

echo "=== [2/2] Gate 3/4: ${STAGE2_TAG} ==="
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

echo "=== Section 53 visualization + Gate 3/4 complete ==="
