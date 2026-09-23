#!/bin/zsh
# User-directed 2026-09-04: "run the visualizations and gate 3/4
# diagnostics for section 85" -- Section 85 (= Section 81 with lambda_z
# removed, w_var x2, w_spatial x1.2, +w_smooth=0.00075). Stage 2 finished:
# best_val_kmax_mse=0.027869, slightly better than Section 81's 0.028618
# at the identical curriculum, cond#=24.04 (healthy), and
# scripts/analyze_latent_smoothness.py showed 85 smoother than 81 on
# every axis measured (step size, curvature, propagator Jacobian norm).
# Same pattern as section81_viz_gate34.sh/section82_viz_gate34.sh.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section85_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_wvar002_wspatial0012_wsmooth00075_nolambdaz_200ep
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

echo "=== Section 85 visualization + GIF + Gate 3/4 complete ==="
