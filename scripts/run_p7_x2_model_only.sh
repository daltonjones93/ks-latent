#!/bin/bash
# P7 follow-up (2026-10-04, user): model-only 2D-KS evaluation at 88^2 (scale 2), seed 0, no truth
# (expected ~330-345 from extensivity of the 44^2/66^2 truths). 512 of 768 directions (D_KY ~340).
# Runs after P9's evaluations (or if the main queue exits), at background QoS.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments
until [ -f $E/crop/p9_a005_x3_seed0.json ] || ! pgrep -f run_c19a_p9_p8 > /dev/null; do sleep 120; done
[ -f $E/crop/p7_ks2d_x2.0_seed0.json ] && exit 0
echo "[p7x2 $(date '+%m-%d %H:%M')] start"
taskpolicy -b python scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_p7_ks2d.pt \
  --prop-ckpt $A/stage2_prop_kolmogorov_crop_p7_ks2d.pt --scale 2.0 --seed 0 --n-directions 512 \
  --tag p7_ks2d_x2.0_seed0 >> $A/logs/crop_eval_p7_ks2d.log 2>&1 && echo "[p7x2] done" || echo "[p7x2] FAILED"
