#!/bin/bash
# R4 reach test (user 2026-10-12, "can we reach 6 sizes per step using 66^2 data?"): r4e showed 4x more 44^2 data
# does not close D's filter gap (0.393 vs 0.408; Paper II model 0.324). Next hypothesis: the propagator reach.
# D has reach 3 sites (16.5 grid units at 44^2); the Paper II model reaches further. Train c = 12, noise 0.1 models on
# the larger boxes with a longer reach that still does not wrap (reach < half the box):
#   r4f: 66^2 box (12 sites), window 5 x 1 layer = reach 5;  r4g: 77^2 box (14 sites), window 6 x 1 = reach 6.
# Then the 44^2 filter (RTPS 0.7, 0.9) and D_KY at 44^2/66^2. Log: artifacts/logs/r4.log, r4_da.log.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; E=$A/experiments
log() { echo "$(date '+%F %T') [R4-F] $*" >> $L/r4.log; }
dlog() { echo "$(date '+%F %T') [R4-DA] $*" >> $L/r4_da.log; }
run() { TAG=$1; DS=$2; CROP=$3; WIN=$4
  if [ ! -f $A/stage2_prop_kolmogorov_crop_$TAG.pt ]; then log "train $TAG start (box $DS crop $CROP window $WIN)"
    $PY scripts/train_kolmogorov_crop.py --tag $TAG --dataset $DS --n-offsets 16 --site-mix-radius 0 --n-site-mix-layers 0 \
      --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000 \
      --crop-sites $CROP --crop-sites-y $CROP --window $WIN --prop-layers 1 --local-channels 12 --data-noise 0.1 > $L/crop_train_$TAG.log 2>&1
    log "train $TAG rc=$? $(grep -E 'step 9000/9000' $L/crop_train_$TAG.log | tail -1)"; fi
  o=$($PY -c "import torch; ck=torch.load('$A/stage1_ae_kolmogorov_crop_$TAG.pt', map_location='cpu', weights_only=False); print(round(0.1 * 6.172126667325263 / ck['normalization_std'], 5))")
  for r in 0.7 0.9; do out=R4_${TAG}_rtps${r}_L44_T4; [ -f $E/da2d/h9_$out.json ] && continue
    OMP_NUM_THREADS=4 $PY scripts/exp_h9_tiled_da.py --tag $TAG --L 44 --n-tilings 4 --rtps $r --obs-stride 8 --obs-noise-std $o --out $out > $L/h9_$out.log 2>&1
    dlog "$out rc=$? $($PY scripts/r4_summary.py da $E/da2d/h9_$out.json)"; done
  for sc in 1.0 1.5; do out=$E/crop/${TAG}_x${sc}_seed0.json; [ -f $out ] && continue
    $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$TAG.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$TAG.pt \
      --scale $sc --seed 0 --tag ${TAG}_x${sc}_seed0 >> $L/crop_eval_$TAG.log 2>&1
    log "eval $TAG x$sc seed0 rc=$? $($PY scripts/r4_summary.py dky $out)"; done; }
run r4f_ks2d_s66_r5_c12d10 $A/datasets/ks2d_box66_large.h5 12 5
run r4g_ks2d_s77_r6_c12d10 $A/datasets/ks2d_box77_large.h5 14 6
log "R4-F DONE"
