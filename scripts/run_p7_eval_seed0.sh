#!/bin/bash
# P7 step 2, reduced (2026-10-04, user): one eval seed per size, background QoS (efficiency cores).
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments
while pgrep -f "train_kolmogorov_crop.py.*p7_ks2d" > /dev/null; do sleep 60; done
[ -f $A/stage2_prop_kolmogorov_crop_p7_ks2d.pt ] || { echo "[p7e] no prop ckpt; abort"; exit 1; }
for sc in 1.0 1.5; do
  out=$E/crop/p7_ks2d_x${sc}_seed0.json
  [ -f $out ] && continue
  echo "[p7e $(date '+%m-%d %H:%M')] eval x$sc seed0"
  taskpolicy -b python scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_p7_ks2d.pt \
    --prop-ckpt $A/stage2_prop_kolmogorov_crop_p7_ks2d.pt --scale $sc --seed 0 --tag p7_ks2d_x${sc}_seed0 \
    >> $A/logs/crop_eval_p7_ks2d.log 2>&1 || echo "[p7e] FAILED x$sc"
done
echo "[p7e] ALL DONE"
