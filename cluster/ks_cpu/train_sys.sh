#!/bin/bash
# Propagator trainings on the frozen Paper I/II autoencoders, one system and one block variant, seeds 0-2 in sequence (skips finished ones).
#   sbatch ... cluster/ks_cpu/train_sys.sh SYSTEM VARIANT      SYSTEM: kolm | ks2d | ks1d     VARIANT: gelu_ln | swiglu_rms
set -uo pipefail
. "$HOME/latent_DA/cluster/ks_cpu/common.sh"
sys=$1; v=$2
for seed in 0 1 2; do
  tag=mx_${sys}_${v}_s${seed}
  if [ "$sys" = ks1d ]; then
    [ -f $A/stage2_prop_patched_full_$tag.pt ] && { echo "skip $tag"; continue; }
    s0=$(date +%s)
    python scripts/train_stage2_patched.py --ae-checkpoint $ks1d_ae --backbone local_mlp --prop-n-tokens 16 --prop-token-d-model 32 --prop-token-n-layers 2 \
      --attn-window 3 --mode markovian --mixer-variant $v --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
      --seed $seed --device cpu --tag $tag > logs/ks_train_$tag.log 2>&1
  else
    [ -f $A/stage2_prop_kolmogorov_crop_$tag.pt ] && { echo "skip $tag"; continue; }
    s0=$(date +%s)
    train_crop $sys $tag $v $seed > logs/ks_train_$tag.log 2>&1
  fi
  echo "TRAIN $tag rc=$? $(( $(date +%s) - s0 )) s: $(grep -E 'step [0-9]+/[0-9]+|epoch' logs/ks_train_$tag.log | tail -1 | cut -c1-150)"
done
