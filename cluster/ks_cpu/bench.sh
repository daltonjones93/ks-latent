#!/bin/bash
# time per Stage B step on this node for the two 2D systems, and 2 epochs of the 1D Stage 2 (leaves small bench_* files in artifacts/)
set -uo pipefail
. "$HOME/latent_DA/cluster/ks_cpu/common.sh"
for sys in kolm ks2d; do
  for v in gelu_ln swiglu_rms; do
    s=$(date +%s); STEPS=60 train_crop $sys bench_${sys}_$v $v 0 2>&1 | grep -E "Stage B:|step 60/60|Error|error" | cut -c1-160
    echo "BENCH $sys $v: $(( $(date +%s) - s )) s for 60 steps incl. data load"
  done
done
for v in gelu_ln swiglu_rms; do
  s=$(date +%s)
  python scripts/train_stage2_patched.py --ae-checkpoint $ks1d_ae --backbone local_mlp --prop-n-tokens 16 --prop-token-d-model 32 --prop-token-n-layers 2 --attn-window 3 --mode markovian --mixer-variant $v --epochs 2 --k-max 12 --k-warmup-epochs 1 --k-mid 8 --k-mid-epochs 1 --tag bench_1d_$v 2>&1 | grep -E "epoch|Error" | tail -2 | cut -c1-160
  echo "BENCH ks1d $v: $(( $(date +%s) - s )) s for 2 epochs incl. load"
done
