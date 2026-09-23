#!/bin/zsh
# User-directed 2026-09-03: "it seems based on this that a FNO based MLP
# might work better for the propagator. can we code this up and test it
# on a stage 2 training using the AE from 65?" -- follow-up to Section
# 66's finding (docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md) that the latent
# index already shows genuine periodic structure under w_spatial_signed
# training, and to the vit-propagator attempt on this same AE plateauing
# at val_kmax_mse~0.22 (much worse than mlp/markovian's 0.0885 or
# mlp/history5's 0.0684 on the same AE).
#
# Implemented a NEW backbone='fno_mlp' (ks_latent/models/propagator.py
# _FNOMLPDeltaBody, ks_latent/config.py PropagatorConfig, wired into
# scripts/train_stage2_patched.py's --backbone choices): FNO spectral-conv
# layers (global, low-frequency mixing across the circular latent index)
# followed by per-token TokenMLPBlock stack (NEW public class in
# autoencoder_vit.py) -- NO attention, NO positional encoding anywhere.
# Verified directly (not just unit-tested): identity-at-init (delta=0 at
# init) AND exact translation-equivariance (perturbed the zero-init output
# head, confirmed step_one(roll(z)) == roll(step_one(z)) to float
# precision) -- see PropagatorConfig's backbone="fno_mlp" docstring for
# the full rationale (this mirrors the KS PDE's own spatial homogeneity).
# 7 new unit tests added (tests/unit/test_propagator_fno_mlp.py), full
# suite re-run clean.
#
# Sizing: n_tokens=4, token_d_model=64, token_n_layers=2 (MLP block
# count), fno_n_layers=2, fno_modes=None (default n_tokens//2+1=3) ->
# 125515 params, comparable to the earlier vit attempt's ~124k for a fair
# comparison.
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section65_mlpmarkovian_wvar0015_wspatialsigned0035_logdet0005_lambdaz0001fromepoch0_200ep.pt
TAG=section65_ae_fnomlp_prop_k12_300ep

echo "=== Stage 2: fno_mlp propagator (FNO spectral-conv + per-token MLP, no attention, translation-equivariant) on Section 65's AE, --amp, k_max=12, 300 epochs ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone fno_mlp --mode markovian \
  --prop-n-tokens 4 --prop-token-d-model 64 --prop-token-n-layers 2 \
  --fno-n-layers 2 \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$TAG" \
  > artifacts/logs/stage2_${TAG}.log 2>&1

echo "=== Section 65 fno_mlp-propagator run complete ==="
tail -3 artifacts/logs/stage2_${TAG}.log
