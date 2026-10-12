#!/bin/bash
# R4 (user 2026-10-12): D on more data. D (c = 12, noise 0.1, 44^2 box, reach 3) filtered at 0.395-0.408 at 44^2 vs
# 0.324 for the Paper II model, with an equally good decoder; RTPS 0.8-1.0 fixed the calibration but not the error.
# The Paper II model saw ~6x more data (264^2 windows = 144 box-areas x 4 trajectories); here D is retrained on 144
# independent 44^2 trajectories (scripts/gen_ks2d_box_large.py), default batches. Log: artifacts/logs/r4.log.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; E=$A/experiments; DS=$A/datasets/ks2d_box44_large.h5; TAG=r4e_ks2d_s44big_r3_c12d10
log() { echo "$(date '+%F %T') [R4-E] $*" >> $L/r4.log; }
dlog() { echo "$(date '+%F %T') [R4-DA] $*" >> $L/r4_da.log; }
if [ ! -f $DS ]; then log "generating $DS"; $PY scripts/gen_ks2d_box_large.py --n-train 144 --n-val 6 --workers 8 --out $DS >> $L/gen_box44_large.log 2>&1; log "data rc=$? $(tail -1 $L/gen_box44_large.log)"; fi
if [ ! -f $A/stage2_prop_kolmogorov_crop_$TAG.pt ]; then
  log "train $TAG start"
  $PY scripts/train_kolmogorov_crop.py --tag $TAG --dataset $DS --n-offsets 16 --site-mix-radius 0 --n-site-mix-layers 0 \
    --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000 \
    --crop-sites 8 --crop-sites-y 8 --window 3 --prop-layers 1 --local-channels 12 --data-noise 0.1 > $L/crop_train_$TAG.log 2>&1
  log "train $TAG rc=$? $(grep -E 'step 9000/9000' $L/crop_train_$TAG.log | tail -1)"
fi
o=$($PY -c "import torch; ck=torch.load('$A/stage1_ae_kolmogorov_crop_$TAG.pt', map_location='cpu', weights_only=False); print(round(0.1 * 6.172126667325263 / ck['normalization_std'], 5))")
for r in 0.7 0.9; do out=R4_${TAG}_rtps${r}_L44_T4; [ -f $E/da2d/h9_$out.json ] && continue
  OMP_NUM_THREADS=4 $PY scripts/exp_h9_tiled_da.py --tag $TAG --L 44 --n-tilings 4 --rtps $r --obs-stride 8 --obs-noise-std $o --out $out > $L/h9_$out.log 2>&1
  dlog "$out rc=$? $($PY scripts/r4_summary.py da $E/da2d/h9_$out.json)"; done
for s in 0 1; do for sc in 1.0 1.5 2.0; do out=$E/crop/${TAG}_x${sc}_seed$s.json; [ -f $out ] && continue
  nd=""; [ $sc = 2.0 ] && nd="--n-directions 512"
  $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$TAG.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$TAG.pt \
    --scale $sc --seed $s --tag ${TAG}_x${sc}_seed$s $nd >> $L/crop_eval_$TAG.log 2>&1
  log "eval $TAG x$sc seed$s rc=$? $($PY scripts/r4_summary.py dky $out)"; done; done
until [ -f $E/da2d/h9_R4_r4d_ks2d_s44_r3_c12d10_L440_T4.json ]; do sleep 60; done
out=R4_${TAG}_L440_T4; [ -f $E/da2d/h9_$out.json ] || { OMP_NUM_THREADS=4 $PY scripts/exp_h9_tiled_da.py --tag $TAG --L 440 --n-tilings 4 --rtps 0.7 --obs-stride 8 --obs-noise-std $o --out $out > $L/h9_$out.log 2>&1; dlog "$out rc=$? $($PY scripts/r4_summary.py da $E/da2d/h9_$out.json)"; }
log "R4-E DONE (ungated 2026-10-12 15:10: no wait for the train lane)"
