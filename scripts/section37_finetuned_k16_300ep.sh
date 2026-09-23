#!/bin/zsh
# User-directed 2026-08-31: "set up another run use AE from
# stage1_ae_patched_full_history3_wspatial005_finetune_from_vitaux_200ep.pt,
# use an mlp propagator, train out to k = 16 at 300 epochs. then run gate 3
# and 4 diagnostics and visualization."
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_history3_wspatial005_finetune_from_vitaux_200ep.pt
TAG=history3_wspatial005_finetune_from_vitaux_200ep_mlpmarkovian_k16_300ep
PROP=artifacts/stage2_prop_patched_full_${TAG}.pt

echo "=== [1/3] Stage 2: fine-tuned AE, fresh mlp/markovian propagator, 300 epochs, k_max=16 ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone mlp --mode markovian \
  --epochs 300 --k-max 16 --k-warmup-epochs 210 --k-mid 10 --k-mid-epochs 175 \
  --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage2_${TAG}.log 2>&1

echo "=== [2/3] Gate 3/4: ${TAG} ==="
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

echo "=== [3/3] Visualization ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$TAG" > artifacts/logs/visualize_${TAG}.log 2>&1

echo "=== Section 37 (fine-tuned AE, fresh mlp propagator, k=16, 300 epochs) complete ==="
