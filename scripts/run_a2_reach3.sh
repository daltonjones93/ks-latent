#!/bin/bash
# A2 (Paper I referee 2, 2026-10-13): separate data from reach in the Kolmogorov box-vs-window contrast.
# Window-trained alpha = 0.05 model with reach 3 (prop-layers 1 x window 3), same Stage A as c19_e0d0_c6
# (copied, so only the propagator's reach differs), evaluated at 1/2/3x on ICs 0-2 like c19_e0d0_c6.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments; PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
TAG=a2_c19_e0d0_c6_r3
log () { echo "$(date '+%F %T') [A2] $*" >> $A/logs/r4.log; }
[ -f $A/stage1_ae_kolmogorov_crop_$TAG.pt ] || cp $A/stage1_ae_kolmogorov_crop_c19_e0d0_c6.pt $A/stage1_ae_kolmogorov_crop_$TAG.pt
log "train $TAG start"
$PY scripts/train_kolmogorov_crop.py --dataset $A/datasets/kolmogorov_wide16_alpha0.05.h5 --tag $TAG \
  --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --local-channels 6 \
  --prop-layers 1 --window 3 > $A/logs/crop_train_$TAG.log 2>&1
log "train $TAG rc=$? $(tail -2 $A/logs/crop_train_$TAG.log | head -1 | cut -c1-120)"
ev () { sc=$1; s=$2; ta=""; [ $sc = 3 ] && [ -f $E/crop/truth_a005_x3_seed$s.json ] && ta="--truth-json $E/crop/truth_a005_x3_seed$s.json"
  $PY scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$TAG.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$TAG.pt \
    --scale $sc --seed $s --tag ${TAG}_x${sc}_seed$s $ta >> $A/logs/crop_eval_$TAG.log 2>&1
  log "eval $TAG x$sc seed$s rc=$? $(tail -1 $A/logs/crop_eval_$TAG.log | cut -c1-140)"; }
for s in 0 1 2; do ev 2 $s & ev 3 $s & wait; done
for s in 0 1 2; do ev 1 $s; done
log "A2 DONE"
