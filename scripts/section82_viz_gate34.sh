#!/bin/zsh
# User-directed 2026-09-04: "run diagnostics and visualizations for 82"
# -- Section 82 (ViT+FourierIFFTBody hybrid, d_latent=56, w_spatial=0.015,
# lambda_z=0.0003, otherwise Section 81's architecture/recipe). Stage 2
# finished: best_val_kmax_mse=0.084004 (k_max=12), notably worse than
# Section 81's 0.028618 at the same curriculum -- the smoothness push
# (higher w_spatial/lambda_z) combined with the larger d_latent=56
# appears to have hurt rollout fit here, mirroring the earlier
# Section 76/77 pattern where pushing w_spatial up degraded Stage-2
# learnability. Same pattern as section81_viz_gate34.sh.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section82_dlatent56_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_wspatial015_lambdaz0003_200ep
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

echo "=== Section 82 visualization + GIF + Gate 3/4 complete ==="
