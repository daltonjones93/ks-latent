#!/bin/bash
# v2 (2026-10-05, user-directed): put the spectral regularizer directly into k_xs005's
# own (only) training run, rather than as a separate k_xsr005 arm. Motivation: nearly
# every Kolmogorov model tonight underestimates D_KY at 2x/3x (P2, c19_e0d0_c6, k_a005,
# p9b_a005, k_xs0025 -- see chat discussion, 2026-10-05), consistent with the classic
# MSE-autoregressive "damping"/variance-collapse bias in chaotic surrogates, worsened by
# long free rollouts. The regularizer directly penalizes this. k_xs0025 (alpha=0.025,
# no regularizer) already gives the baseline-without-regularizer comparison point at the
# harder drag; this run (alpha=0.05, WITH regularizer from the start) is the fix test.
# x-strip training (exactly c19_e0d0_c6's flags) + P9b long rollout (k_max=32), alpha=0.05.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments; D=$A/datasets/kolmogorov_wide16_alpha0.05.h5
T3=$E/crop/truth_a005_x3_seed0.json
log () { echo "[kxs005 $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }
ev () { tag=$1; sc=$2; s=$3; ta=""; [ $sc = 3 ] && ta="--truth-json $T3"
  run $E/crop/${tag}_x${sc}_seed$s.json "$tag eval x$sc seed$s" sh -c "python scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s $ta >> $A/logs/crop_eval_$tag.log 2>&1"; }

BASE="--site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 \
--local-channels 6 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000"

run $A/stage2_prop_kolmogorov_crop_k_xs005.pt "train k_xs005 (x-strip, long rollout, + spectral reg)" sh -c \
  "python scripts/train_kolmogorov_crop.py --dataset $D --tag k_xs005 $BASE --w-spectral 1.0 \
   >> $A/logs/crop_train_k_xs005.log 2>&1"
for sc in 1 2 3; do for s in 0 1 2; do ev k_xs005 $sc $s; done; done

log "ALL DONE"
