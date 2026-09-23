#!/bin/zsh
# User-directed 2026-09-05 (standing "visualizations + Gate 3/4" practice,
# applied to Section 95 despite its rollout-accuracy regression: "I think
# this may still be easier to learn a pde to model the dynamics" -- the
# user wants the full structural/smoothness picture regardless of raw
# accuracy, since PDE-readiness and rollout accuracy are established as
# different axes throughout this project). Section 95 (= Section 94 +
# w_spatial 0.019->0.03, w_smooth 0.0005->0.0015) regressed on
# val_kmax_mse (0.0198 vs 94's 0.0123) and did NOT improve smoothness
# (flat-to-slightly-worse on both the encoder-side and propagator-Jacobian
# measures) -- Gate 3/4 will show whether it nonetheless has different
# structural properties (D_KY, D3/D4/D8) worth knowing about. Same
# pattern as section94's own (never run) counterpart.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section95_vitfourieriffthybrid_dmodel64_fourierhidden95_tokenmlp_propmlpmarkovian52_wspatial03_wsmooth0015_200ep
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

echo "=== [3/3] Gate 3/4 (incl. D9 smoothness): ${STAGE2_TAG} ==="
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

echo "=== Section 95 visualization + GIF + Gate 3/4 (incl. D9) complete ==="
