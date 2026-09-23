#!/bin/zsh
# User-directed 2026-08-31: "run it with the vit with global attention in
# stage 2 with global attention. no output nonlinearity" -- continuing
# Section 43's propagator-architecture comparison (masked_mlp attempts at
# attn_window=2 and 4 were both killed early) on the same AE
# (wspatial085_logdet0001_120ep, the strongest measured spatial coherence
# of anything tried: l_spatial=0.368, D7 p=0.0000), now with a vit
# backbone and NO --attn-window (default None = full/global attention,
# the opposite locality bet from masked_mlp) and NO --delta-cap (default
# None = no output nonlinearity/tanh squashing on the residual).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_history3_fullprop_wvar005_tw16_wspatial085_logdet0001_120ep.pt
TAG=history3_fullprop_wvar005_tw16_wspatial085_logdet0001_120ep_vitglobal_k12_300ep

echo "=== [1/2] Stage 2: vit backbone, global attention (no attn-window), no delta_cap, k_max=12, 300 epochs ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone vit --pos-encoding linear --mode markovian \
  --prop-n-tokens 44 --prop-token-d-model 64 \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$TAG" \
  > artifacts/logs/stage2_${TAG}.log 2>&1

PROP=artifacts/stage2_prop_patched_full_${TAG}.pt

echo "=== [2/2] Gate 3/4: ${TAG} ==="
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

echo "=== Section 43 (vit, global attention, no output nonlinearity) complete ==="
tail -3 artifacts/logs/stage2_${TAG}.log
