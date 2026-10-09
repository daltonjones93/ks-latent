#!/bin/bash
# Clean A/B companion to k_xs005 (2026-10-06, user-approved): identical recipe --
# x-strip training (c19_e0d0_c6's flags), P9b long rollout (k_max=32), alpha=0.05 --
# with the ONLY difference being --w-spectral 0 (off) instead of 1.0. Isolates the
# regularizer's actual contribution from the alpha=0.05-vs-0.025 confound flagged in
# sec:k_xs005_full.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments; D=$A/datasets/kolmogorov_wide16_alpha0.05.h5
T3=$E/crop/truth_a005_x3_seed0.json
log () { echo "[kxs005nr $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }
ev () { tag=$1; sc=$2; s=$3; ta=""; [ $sc = 3 ] && ta="--truth-json $T3"
  run $E/crop/${tag}_x${sc}_seed$s.json "$tag eval x$sc seed$s" sh -c "python scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s $ta >> $A/logs/crop_eval_$tag.log 2>&1"; }

BASE="--site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 \
--local-channels 6 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000"

run $A/stage2_prop_kolmogorov_crop_k_xs005_noreg.pt "train k_xs005_noreg (x-strip, long rollout, NO reg)" sh -c \
  "python scripts/train_kolmogorov_crop.py --dataset $D --tag k_xs005_noreg $BASE \
   >> $A/logs/crop_train_k_xs005_noreg.log 2>&1"
for sc in 1 2 3; do for s in 0 1 2; do ev k_xs005_noreg $sc $s; done; done

log "ALL DONE"
