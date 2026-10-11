#!/bin/bash
# KDA (2026-10-11, user-approved): noisy-trained 12-channel Kolmogorov model for the Paper II Kolmogorov DA,
# the Part II KS recipe (c = 12, data noise 0.1, k_max 32) on the canonical alpha = 0.05 x-strip setup
# (k_xs005_noreg flags, no spectral regularizer). Starts after the three J17G trainings (one MPS training
# at a time), then evaluates D_KY at 1x/2x/3x, seed 0, against the alpha = 0.05 truths.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; E=$A/experiments; D=$A/datasets/kolmogorov_wide16_alpha0.05.h5; TAG=k_xs005_c12d10
T3=$E/crop/truth_a005_x3_seed0.json; LOG=$A/logs/kda.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
until grep -q "J17G_d0.3\] training done" $A/logs/j17g.log 2>/dev/null; do sleep 300; done
while pgrep -f "train_kolmogorov_crop.py" > /dev/null; do sleep 120; done
log "training $TAG start"
[ -f $A/stage2_prop_kolmogorov_crop_$TAG.pt ] || $PY scripts/train_kolmogorov_crop.py --dataset $D --tag $TAG \
  --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 \
  --local-channels 12 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000 --data-noise 0.1 \
  > $A/logs/crop_train_$TAG.log 2>&1
log "training done rc=$?"
for sc in 1 2 3; do ta=""; [ $sc = 3 ] && ta="--truth-json $T3"
  [ -f $E/crop/${TAG}_x${sc}_seed0.json ] || $PY scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$TAG.pt \
    --prop-ckpt $A/stage2_prop_kolmogorov_crop_$TAG.pt --scale $sc --seed 0 --tag ${TAG}_x${sc}_seed0 $ta \
    >> $A/logs/crop_eval_$TAG.log 2>&1
  log "eval x$sc rc=$? $(grep -E 'rel_err_pct' $A/logs/crop_eval_$TAG.log | tail -1)"
done
log "KDA TRAIN DONE"
