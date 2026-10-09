#!/bin/bash
# P9 follow-up (2026-10-04): IC seeds 1,2 at 3x (seed 0 gave 42.6%), normal priority (~5 min each).
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments
for s in 1 2; do
  [ -f $E/crop/p9_a005_x3_seed$s.json ] && continue
  python scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_p9_a005.pt \
    --prop-ckpt $A/stage2_prop_kolmogorov_crop_p9_a005.pt --scale 3 --seed $s --tag p9_a005_x3_seed$s \
    --truth-json $E/crop/truth_a005_x3_seed0.json > $A/logs/p9_x3_seed$s.log 2>&1 && echo "[p9s] seed $s done" || echo "[p9s] seed $s FAILED"
done
