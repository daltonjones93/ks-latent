#!/bin/zsh
# User-directed 2026-09-05 (standing "visualizations + Gate 3/4" practice,
# applied to Section 96). Section 96 (= Section 95 with w_smooth
# disabled entirely and w_spatial pushed further to 0.04) regressed
# further than 95 on val_kmax_mse (0.029794 vs 95's 0.0198) and
# conditioning (cond#=160 vs 95's 110), confirming w_spatial alone (not
# w_smooth) drives the accuracy/conditioning cost as it's pushed past
# Section 93/94's sweet spot (w_spatial=0.019). Same pattern as
# section95_viz_gate34.sh.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section96_vitfourieriffthybrid_dmodel64_fourierhidden95_tokenmlp_propmlpmarkovian52_wspatial04_wsmooth0_200ep
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

echo "=== Section 96 visualization + GIF + Gate 3/4 (incl. D9) complete ==="
