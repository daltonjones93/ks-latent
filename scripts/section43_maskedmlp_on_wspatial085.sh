#!/bin/zsh
# User-directed 2026-08-31: "could we try stage 2 using this autoencoder
# [stage1_history3_fullprop_wvar005_tw16_wspatial085_logdet0001_120ep] with
# the masked mlp. let it run 300 epochs. then run gate 3 and gate 4 on the
# result." -- direct test of Section 42's PDE/local-stencil idea: this AE
# has the strongest measured spatial coherence of anything tried
# (l_spatial=0.368, D7 bandedness=0.5113, p=0.0000), so a locally-masked
# propagator (masked_mlp, --attn-window 2 -- the same precedented window
# used for maskedmlp_encoder_e2e, Sections 17/22/23) has real structure to
# exploit, unlike earlier masked_mlp tests run on AEs with no confirmed
# locality.
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_history3_fullprop_wvar005_tw16_wspatial085_logdet0001_120ep.pt
TAG=history3_fullprop_wvar005_tw16_wspatial085_logdet0001_120ep_maskedmlp_w2_k12_300ep

echo "=== [1/2] Stage 2: masked_mlp (attn_window=2), w_varmatch=0, k_max=12, 300 epochs ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone masked_mlp --attn-window 2 --mode markovian \
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

echo "=== Section 43 (masked_mlp on the most spatially-coherent AE) complete ==="
tail -3 artifacts/logs/stage2_${TAG}.log
