#!/bin/zsh
# User-directed 2026-09-03: "can we rerun stage 2 training with 65 warm
# starting using the existing propagator, please run it for another 400
# epochs up to k = 16." -- Section 65's val_kmax_mse (0.088522) was by far
# the worst in the sweep despite completely healthy Lyapunov spectrum
# (D_KY=21.99, essentially matching the true ~22) and solid DA cycling
# skill (skill_free_over_da=3.32); this continues training longer with a
# harder k-curriculum (12 -> 16) to see whether extended Stage 2 training
# fixes the free-rollout compounding-error problem the healthy dynamics
# suggest should be fixable.
# Warm-starts from Section 65's own ALREADY-TRAINED Stage 2 propagator
# (not the Stage 1 aux propagator, unlike every other warm-start in this
# document) via --init-prop-checkpoint.
# k-curriculum scaled using this project's standard slow-curriculum
# convention (k_warmup_epochs ~ 0.7*epochs, k_mid ~ 2/3*k_max, k_mid_epochs
# ~ 5/6*k_warmup_epochs -- same ratios as the canonical 300ep/k_max=12
# recipe's 210/8/175): epochs=400 -> k_warmup_epochs=280, k_max=16 ->
# k_mid=11, k_mid_epochs=233.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section65_mlpmarkovian_wvar0015_wspatialsigned0035_logdet0005_lambdaz0001fromepoch0_200ep
AE=artifacts/stage1_ae_patched_full_${TAG}.pt
EXISTING_PROP=artifacts/stage2_prop_patched_full_${TAG}_warmstart_k12_300ep.pt
STAGE2_TAG="${TAG}_warmstart_k12_300ep_continued_k16_400ep"

echo "=== Stage 2 continued: warm-started from Section 65's existing (k_max=12/300ep) Stage 2 propagator, --amp, k_max=16, 400 epochs ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$EXISTING_PROP" \
  --amp \
  --epochs 400 --k-max 16 --k-warmup-epochs 280 --k-mid 11 --k-mid-epochs 233 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 65 continued Stage 2 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
