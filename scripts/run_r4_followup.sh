#!/bin/bash
# R4 follow-up (2026-10-12): the Paper II recipe (c = 12, data noise 0.1) on the 44^2 data with run C's short-reach
# propagator, so the small-source filter can be compared with Paper II's 12-channel model (decoder floor 0.25 vs
# 0.66-0.69 for every 6-channel model, which is a channel-count effect, not a data effect). Starts after the main R4
# queue; D_KY at 44^2/66^2/88^2 and the filter at 44^2 and 440^2. Log: artifacts/logs/r4.log and r4_da.log.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; E=$A/experiments; D=$A/datasets; TAG=r4d_ks2d_s44_r3_c12d10
log() { echo "$(date '+%F %T') [R4-D] $*" >> $L/r4.log; }
until grep -q "R4 DONE" $L/r4.log; do sleep 120; done
if [ ! -f $A/stage2_prop_kolmogorov_crop_$TAG.pt ]; then
  log "train $TAG start"
  $PY scripts/train_kolmogorov_crop.py --dataset $D/ks2d_trajectories_default.h5 --tag $TAG --n-offsets 16 \
    --local-channels 12 --data-noise 0.1 --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 \
    --dec-n-site-mix-layers 0 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000 \
    --crop-sites 8 --crop-sites-y 8 --window 3 --prop-layers 1 --ae-batch 800 --prop-batch 1600 \
    > $L/crop_train_$TAG.log 2>&1
  log "train $TAG rc=$? $(grep -E 'step 9000/9000' $L/crop_train_$TAG.log | tail -1)"
fi
for s in 0 1; do for sc in 1.0 1.5 2.0; do out=$E/crop/${TAG}_x${sc}_seed$s.json; [ -f $out ] && continue
  nd=""; [ $sc = 2.0 ] && nd="--n-directions 512"
  $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$TAG.pt \
    --prop-ckpt $A/stage2_prop_kolmogorov_crop_$TAG.pt --scale $sc --seed $s --tag ${TAG}_x${sc}_seed$s $nd \
    >> $L/crop_eval_$TAG.log 2>&1
  rc=$?; log "eval $TAG x$sc seed$s rc=$rc $($PY scripts/r4_summary.py dky $out)"
done; done
until grep -q "R4 DA DONE" $L/r4_da.log; do sleep 120; done
o=$($PY -c "import torch; ck=torch.load('$A/stage1_ae_kolmogorov_crop_$TAG.pt', map_location='cpu', weights_only=False); print(round(0.1 * 6.172126667325263 / ck['normalization_std'], 5))")
for Ls in 44 440; do out=R4_${TAG}_L${Ls}_T4; [ -f $E/da2d/h9_$out.json ] && continue
  OMP_NUM_THREADS=4 $PY scripts/exp_h9_tiled_da.py --tag $TAG --L $Ls --n-tilings 4 --rtps 0.7 --obs-stride 8 \
    --obs-noise-std $o --out $out > $L/h9_$out.log 2>&1
  rc=$?; echo "$(date '+%F %T') [R4-DA] $out rc=$rc obs=$o $($PY scripts/r4_summary.py da $E/da2d/h9_$out.json)" >> $L/r4_da.log
done
log "R4 FOLLOWUP DONE"
