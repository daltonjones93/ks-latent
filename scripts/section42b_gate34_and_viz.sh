#!/bin/zsh
# User-directed 2026-09-01: "let it finish, run gate 3 and 4 and
# visualize." Waits on the currently-running Section 42b Stage 2 (vit/
# history(n=2) aux AE, fresh mlp/markovian, no w_spatial, no delta_cap
# confound, k_max=12), then runs Gate 3/4 + visualize_rollout.py.
set -e
cd /Users/daltonjones/Documents/latent_DA

echo "=== waiting for Section 42b Stage 2 (PID 20596) to finish ==="
while kill -0 20596 2>/dev/null; do sleep 10; done

AE=artifacts/stage1_ae_patched_full_section42b_vitaux_h2_wspatial08_wpred1_ms4_logdet0005.pt
TAG=section42b_vitaux_h2_wspatial08_wpred1_ms4_logdet0005_freshmlp_nowspatial_k12
PROP=artifacts/stage2_prop_patched_full_${TAG}.pt

echo "=== [1/2] Gate 3/4: ${TAG} ==="
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

echo "=== [2/2] Visualization: ${TAG} ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$TAG" > artifacts/logs/visualize_${TAG}.log 2>&1

echo "=== Section 42b Gate 3/4 + visualization complete ==="
