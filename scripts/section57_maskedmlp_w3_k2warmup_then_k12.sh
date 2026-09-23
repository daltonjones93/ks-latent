#!/bin/zsh
# User-directed 2026-09-02: "I want to run 400 epochs at k = 2, then
# continue onto the same stage 2 training as before with the same
# parameters I specified before." -- two-phase Stage 2 on Section 52's
# AE, masked_mlp propagator, --attn-window 3 (same architecture as the
# earlier Section 57 attempt, which was killed before this two-phase plan
# was decided):
#   Phase A: from scratch, k_max=2 (k_min is also 2, so this holds k=2 for
#     the entire run -- no curriculum ramp), 400 epochs.
#   Phase B: warm-started via --init-prop-checkpoint from Phase A's own
#     output (preserves the exact masked_mlp/attn_window=3 architecture,
#     ignoring --backbone/--attn-window on this second call), then the
#     same k_max=12 curriculum/epoch count as the original (killed)
#     Section 57 attempt: k_max=12, k_warmup_epochs=210, k_mid=8,
#     k_mid_epochs=175, 300 epochs.
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section52_mlpmarkovian_wvar002_wspatialsigned01_logdet0035_200ep.pt
TAG_A=section52_mlpmarkovian_wvar002_wspatialsigned01_logdet0035_200ep_maskedmlp_w3_k2_400ep
TAG_B=section52_mlpmarkovian_wvar002_wspatialsigned01_logdet0035_200ep_maskedmlp_w3_k2warmup400ep_then_k12_300ep

echo "=== [1/2] Stage 2 Phase A: masked_mlp (attn_window=3), from scratch, --amp, k=2 fixed, 400 epochs ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone masked_mlp --attn-window 3 --mode markovian \
  --amp \
  --epochs 400 --k-max 2 \
  --tag "$TAG_A" \
  > artifacts/logs/stage2_${TAG_A}.log 2>&1

PROP_A=artifacts/stage2_prop_patched_full_${TAG_A}.pt

echo "=== [2/2] Stage 2 Phase B: warm-started from Phase A, --amp, k_max=12, 300 epochs ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$PROP_A" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$TAG_B" \
  > artifacts/logs/stage2_${TAG_B}.log 2>&1

echo "=== Section 57 (masked_mlp w3, k2-warmup then k12) complete ==="
tail -3 artifacts/logs/stage2_${TAG_B}.log
