#!/bin/zsh
# User-directed 2026-09-03: "run a stage 2 training with the 65
# autoencoder and a vit propagator. don't use any kind of nonlinearity on
# the output. make it markovian and use global attention. it should have
# about 120000 params." -- exploratory test of a vit-backbone Stage 2
# propagator on Section 65's AE (which had the best D_KY/attractor-fidelity
# match in the whole sweep despite its poor free-rollout val_kmax_mse).
#
# Mapping the request onto PropagatorConfig/CLI flags:
#   "no nonlinearity on the output" -> do NOT set --delta-cap (default
#     None = off). delta_cap is literally the only output nonlinearity a
#     propagator backbone can have here (delta = delta_cap*tanh(raw_delta
#     /delta_cap), see PropagatorConfig's docstring) -- leaving it unset
#     means the raw (linear, zero-init) output head delta is used as-is.
#   "markovian" -> --mode markovian.
#   "global attention" -> leave --attn-window unset (default None = full/
#     global attention, not the local +-window-neighbours restriction).
#   "~120000 params" -> --prop-n-tokens 4 --prop-token-d-model 100
#     --prop-token-nhead 4 --prop-token-n-layers 1 (the last two are NEW
#     CLI flags added today -- token_nhead/token_n_layers were previously
#     hardcoded to 2/2 in this script). Checked directly by instantiating
#     PropagatorConfig/LatentPropagator across a token_d_model/n_layers
#     sweep: this combination gives 123812 params, the closest achievable
#     to 120000 with token_d_model required to divide token_nhead evenly
#     (100/4=25) -- n_layers=2 alternatives only land at 101580 (d_model=64)
#     or 157692 (d_model=80), both farther off.
#
# From scratch (no warm-start), standard recipe: --amp, k_max=12, 300
# epochs (this project's canonical Stage-2 curriculum).
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section65_mlpmarkovian_wvar0015_wspatialsigned0035_logdet0005_lambdaz0001fromepoch0_200ep.pt
TAG=section65_ae_vitprop_markovian_global_nodeltacap_120kparams_k12_300ep

echo "=== Stage 2: vit propagator (markovian, global attention, no delta_cap, ~120k params) on Section 65's AE, --amp, k_max=12, 300 epochs ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone vit --mode markovian \
  --prop-n-tokens 4 --prop-token-d-model 100 --prop-token-nhead 4 --prop-token-n-layers 1 \
  --pos-encoding linear \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$TAG" \
  > artifacts/logs/stage2_${TAG}.log 2>&1

echo "=== Section 65 vit-propagator run complete ==="
tail -3 artifacts/logs/stage2_${TAG}.log
