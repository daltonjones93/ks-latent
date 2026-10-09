#!/bin/bash
# User-directed (2026-10-05): test the spectral-band-share regularizer on Kolmogorov
# alpha=0.05, where we already have a direct, same-dataset baseline (k_a005, canonical
# recipe, k_max=32, no regularizer) that showed a confusing non-monotone pattern across
# sizes (native ~0.9-12.9% depending on seed, 2x 29.5%/pending, 3x 15.1%).
# k_b005 = IDENTICAL recipe to k_a005 EXCEPT: k_max 32->16, k_mid 16->8 (user: "only
# rollout to 16 steps"), plus --w-spectral 1.0 (Kolmogorov-default bands 0,2,3,5).
# Two variables change at once (rollout length AND regularizer) -- this is a deliberate
# test of a specific combined hypothesis (can the regularizer let a SHORTER, cheaper
# rollout match or beat the long-rollout result), not a single-variable ablation of
# k_a005; noted plainly in the tex writeup, same caution as the original P9->P9b change.
#
# PRIORITY: inserted ahead of k_a0025/k_c0025 in the training lane. Polls specifically
# for k_a01's own checkpoint (every 10s, not the other scripts' 60s generic "is anything
# training" poll) so it grabs the GPU as soon as k_a01 finishes, ahead of the main
# consolidation queue's own k_a0025 start and run_k_c0025.sh's wait.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments; D=$A/datasets/kolmogorov_wide6x6_alpha0.05.h5
T3=$E/crop/truth_a005_x3_seed0.json
log () { echo "[kb005 $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }

until [ -f $A/stage2_prop_kolmogorov_crop_k_a01.pt ]; do sleep 10; done
log "k_a01 done; grabbing training lane for k_b005 (priority insert)"

CANON="--crop-sites 40 --crop-sites-y 40 --n-offsets 16 --local-channels 6 --site-mix-radius 0 --n-site-mix-layers 0 \
--dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 16 --k-mid 8 --noise-end 0.02 --prop-steps 9000 --y-align 32"
run $A/stage2_prop_kolmogorov_crop_k_b005.pt "train k_b005 (k_max=16 + spectral reg)" sh -c \
  "python scripts/train_kolmogorov_crop.py --dataset $D --tag k_b005 $CANON --w-spectral 1.0 \
   >> $A/logs/crop_train_k_b005.log 2>&1"

for sc in 1 2 3; do for s in 0 1; do
  ta=""; [ $sc = 3 ] && ta="--truth-json $T3"
  run $E/crop/k_b005_x${sc}_seed$s.json "k_b005 eval x$sc seed$s" sh -c "python scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_k_b005.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_k_b005.pt \
    --scale $sc --seed $s --tag k_b005_x${sc}_seed$s $ta >> $A/logs/crop_eval_k_b005.log 2>&1"
done; done
log "ALL DONE"
