#!/bin/zsh
# User-directed 2026-08-31: "could you implement 3, and test it in stage 2
# with the same AE architecture." Option 3 from the propagator-brainstorm
# was a Neural-ODE-style propagator (dz/dt = f_theta(z), local/
# convolutional f, fixed-step RK4 integration) -- now implemented as
# backbone="node" (ks_latent/models/propagator.py's _NeuralODEDeltaBody/
# _LocalVectorField, ks_latent/config.py's PropagatorConfig.ode_substeps).
# Testing on the same AE as the masked_mlp/vit attempts
# (wspatial085_logdet0001_120ep, the strongest measured spatial coherence
# of anything tried: l_spatial=0.368, D7 bandedness=0.5113, p=0.0000).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_history3_fullprop_wvar005_tw16_wspatial085_logdet0001_120ep.pt
TAG=history3_fullprop_wvar005_tw16_wspatial085_logdet0001_120ep_node_w2_k12_300ep

echo "=== [1/2] Stage 2: node (attn_window=2, ode_substeps=4), w_varmatch=0, k_max=12, 300 epochs ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone node --attn-window 2 --mode markovian \
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

echo "=== Section 43 (node / Neural ODE propagator) complete ==="
tail -3 artifacts/logs/stage2_${TAG}.log
