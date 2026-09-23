#!/bin/zsh
# User-directed 2026-09-05 (standing "visualizations + Gate 3/4" practice,
# applied to Section 98 -- the critical test case for whether D4
# translation equivariance survives removing the Fourier branch entirely
# from the vit_fourier_hybrid architecture). Section 98 (= Section 97
# with the Fourier branch removed, plain ViT only) matched 97 closely on
# accuracy/conditioning (val_kmax_mse=0.033184 vs 97's 0.031086,
# cond#=155.7 vs 161.2, D8=0.7392 vs 0.7346) -- confirming the Fourier
# branch (only ~20% of params) wasn't doing much for those axes. This
# Gate 4 run is the direct test of the D4 hypothesis: every
# vit_fourier_hybrid checkpoint tested (81, 85, 89-97) got "Genuine
# (approximately) equivariant translation representation found", while
# Section 52 (plain ViT + mlp/markovian propagator, no Fourier branch)
# consistently got "No clean linear translation representation found" --
# predicted that removing the Fourier branch here would revert to
# Section 52's verdict.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section98_vitonly_dmodel56_tokenmlp_propmlpmarkovian52_wspatial04_wsmooth003_200ep
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

echo "=== Section 98 visualization + GIF + Gate 3/4 (incl. D9) complete ==="
