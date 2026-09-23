#!/bin/zsh
# User-directed 2026-09-03: "seems like the vit for the 65 propagator
# doesn't work super well. what if we switch to an mlp with significant
# history. like an mlp with history = 5." -- the vit-backbone attempt
# (markovian, global attention, ~124k params, no delta_cap) plateaued at
# val_kmax_mse~0.22 (worse than the mlp/markovian baseline's 0.0885 on
# this same AE), so this tries backbone='mlp' with mode='history'
# (n_history=5: 4 past states + current) instead -- a generalization of
# 'two_step' to an arbitrary history length, per PropagatorConfig's
# mode='history' docstring.
#
# Caught and fixed a real bug while wiring this up: train_stage2_patched.py's
# 'mlp' backbone branch never passed --n-history through to PropagatorConfig
# (only the transformer/vit/fno_vit/local_mlp branch did), so
# --backbone mlp --mode history --n-history 5 would have silently used the
# n_history=3 default instead. Fixed by adding n_history=args.n_history to
# that branch's PropagatorConfig(...) call; verified directly (instantiated
# PropagatorConfig(mode='history', backbone='mlp', n_history=5), confirmed
# prop.cfg.n_history==5 and rollout_history(...) runs with the right shapes).
#
# Default hidden/n_blocks (128/3, the Stage-2 standard sizing) -- no
# specific param-count target given this time, "significant history" is
# the point of this test, not model size.
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section65_mlpmarkovian_wvar0015_wspatialsigned0035_logdet0005_lambdaz0001fromepoch0_200ep.pt
TAG=section65_ae_mlp_history5_k12_300ep

echo "=== Stage 2: mlp backbone, mode=history (n_history=5), on Section 65's AE, --amp, k_max=12, 300 epochs ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone mlp --mode history --n-history 5 \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$TAG" \
  > artifacts/logs/stage2_${TAG}.log 2>&1

echo "=== Section 65 mlp/history5 run complete ==="
tail -3 artifacts/logs/stage2_${TAG}.log
