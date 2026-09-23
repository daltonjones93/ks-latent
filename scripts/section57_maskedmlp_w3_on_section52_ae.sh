#!/bin/zsh
# User-directed 2026-09-02: "can we try a stage 2 training using the
# autoencoder from section 52. please use a masked mlp for a propagator
# (start from scratch) with window size 3." -- Section 52's AE has
# confirmed spatial coherence (D7/D8 both p=0.0000), so this tests
# whether a locally-masked propagator (masked_mlp, --attn-window 3, the
# window size reused as the mask radius per PropagatorConfig's
# backbone="masked_mlp" docstring) can exploit that structure, following
# the same template as Section 43's masked_mlp test (there with
# --attn-window 2 on a different AE). Trained from scratch (no
# --init-prop-checkpoint), --amp per this session's standing practice.
# Unlike Section 43, the user did not ask for Gate 3/4 in the same
# request this time -- Stage 2 training only; visualize/Gate 3/4 to be
# run as a separate follow-up if/when requested.
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section52_mlpmarkovian_wvar002_wspatialsigned01_logdet0035_200ep.pt
TAG=section52_mlpmarkovian_wvar002_wspatialsigned01_logdet0035_200ep_maskedmlp_w3_k12_300ep

echo "=== Stage 2: masked_mlp (attn_window=3), from scratch, --amp, k_max=12, 300 epochs ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone masked_mlp --attn-window 3 --mode markovian \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$TAG" \
  > artifacts/logs/stage2_${TAG}.log 2>&1

echo "=== Section 57 (masked_mlp w3 on Section 52's AE) Stage 2 complete ==="
tail -3 artifacts/logs/stage2_${TAG}.log
