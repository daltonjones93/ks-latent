#!/bin/bash
# v2 (2026-10-04): adds the C19 per-patch c=6 arm after P9. v1: 2026-10-04 (user: option 1). Finish C19 arm c19_e6d0 only (remaining C19 arms dropped),
# then P9 (2D-patch Kolmogorov, alpha=0.05, 6x6-box domain, 40x40-site crops, y-offsets
# aligned to the forcing period = 32 grid points), then the rest of P8.
# Seed-0 evals; everything at background QoS (taskpolicy -b); training on MPS.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments
log () { echo "[q $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }
T3=$E/crop/truth_a005_x3_seed0.json
evals () { tag=$1; for sc in 1 2 3; do ta=""; [ $sc = 3 ] && ta="--truth-json $T3"
  run $E/crop/${tag}_x${sc}_seed0.json "$tag eval x$sc" taskpolicy -b \
    python scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt \
      --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt --scale $sc --seed 0 --tag ${tag}_x${sc}_seed0 $ta \
      >> $A/logs/crop_eval_$tag.log 2>&1; done; }

# C19 arm 1
while pgrep -f "train_kolmogorov_crop.py.*c19_e6d0" > /dev/null; do sleep 60; done
[ -f $A/stage2_prop_kolmogorov_crop_c19_e6d0.pt ] && evals c19_e6d0 || log "c19_e6d0: no prop ckpt"

# P9
D9=$A/datasets/kolmogorov_wide6x6_alpha0.05.h5
while pgrep -f "gen_kolmogorov_wide_dataset.py.*wide6x6" > /dev/null; do sleep 60; done
run $D9 "P9 data" taskpolicy -b python scripts/gen_kolmogorov_wide_dataset.py --alpha 0.05 --x-scale 6 --y-scale 6 \
  --seed-base 900 --out $D9 >> $A/logs/p9_data.log 2>&1
run $A/stage2_prop_kolmogorov_crop_p9_a005.pt "P9 train" taskpolicy -b \
  python scripts/train_kolmogorov_crop.py --dataset $D9 --tag p9_a005 --crop-sites 40 --crop-sites-y 40 \
    --y-align 32 --n-offsets 16 >> $A/logs/crop_train_p9_a005.log 2>&1
evals p9_a005

# C19 per-patch arm with c=6 (2026-10-04, user: after P9, before P8)
run $A/stage2_prop_kolmogorov_crop_c19_e0d0_c6.pt "c19_e0d0_c6 train" taskpolicy -b \
  python scripts/train_kolmogorov_crop.py --dataset $A/datasets/kolmogorov_wide16_alpha0.05.h5 --tag c19_e0d0_c6 \
    --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --local-channels 6 \
    >> $A/logs/crop_train_c19_e0d0_c6.log 2>&1
[ -f $A/stage2_prop_kolmogorov_crop_c19_e0d0_c6.pt ] && evals c19_e0d0_c6

# P8 remainder
D025=$A/datasets/kolmogorov_wide16_alpha0.025.h5
while pgrep -f "gen_kolmogorov_wide_dataset.py --alpha 0.025" > /dev/null; do sleep 60; done
run $D025 "alpha=0.025 wide data" taskpolicy -b python scripts/gen_kolmogorov_wide_dataset.py --alpha 0.025 \
  --out $D025 >> $A/logs/p0_wide_a0.025.log 2>&1
run $A/stage2_prop_kolmogorov_crop_p8_a0025.pt "P8 (alpha=0.025) train" taskpolicy -b \
  python scripts/train_kolmogorov_crop.py --dataset $D025 --tag p8_a0025 >> $A/logs/crop_train_p8_a0025.log 2>&1
for sc in 1 2 3; do for s in 0 1 2; do
  run $E/crop/p8_a0025_x${sc}_seed$s.json "P8 eval x$sc seed$s" taskpolicy -b \
    python scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_p8_a0025.pt \
      --prop-ckpt $A/stage2_prop_kolmogorov_crop_p8_a0025.pt --scale $sc --seed $s --tag p8_a0025_x${sc}_seed$s \
      --truth-json $E/crop/truth_a0p025_x${sc}_seed0.json >> $A/logs/crop_eval_p8.log 2>&1
done; done
log "ALL DONE"
