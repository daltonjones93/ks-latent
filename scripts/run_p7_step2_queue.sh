#!/bin/bash
# P7 step 2 (2026-10-03): 2D-crop training on a wide 6x44 square 2D KS domain with the
# unchanged Section 230 architecture (encoder radius 2x3, propagator window 3):
# 40x40-site crops -> 12-site margin per side -> 16x16-site free interior.
# Evaluate at 44^2 (truth 81.77) and 66^2 (truth 192.5), 3 seeds each.
# Waits for the wide dataset and for the P6 queue; one training job at a time.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments; DATA=$A/datasets/ks2d_wide6.h5
while [ ! -f $DATA ] || pgrep -f run_p6_queue.sh > /dev/null; do sleep 60; done
log () { echo "[p7s2 $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }
run $A/stage2_prop_kolmogorov_crop_p7_ks2d.pt "P7 train (2D crops 40x40)" \
  python scripts/train_kolmogorov_crop.py --dataset $DATA --tag p7_ks2d --crop-sites 40 --crop-sites-y 40 \
    --n-offsets 16 >> $A/logs/crop_train_p7_ks2d.log 2>&1
for sc in 1.0 1.5; do for s in 0 1 2; do
  run $E/crop/p7_ks2d_x${sc}_seed$s.json "P7 eval x$sc seed$s" \
    python scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_p7_ks2d.pt \
      --prop-ckpt $A/stage2_prop_kolmogorov_crop_p7_ks2d.pt --scale $sc --seed $s --tag p7_ks2d_x${sc}_seed$s \
      >> $A/logs/crop_eval_p7_ks2d.log 2>&1
done; done
log "ALL DONE"
