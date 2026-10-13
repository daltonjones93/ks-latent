#!/bin/bash
# R14 (user 2026-10-13): the 2D KS C2/C3 recipes applied to Kolmogorov flow, alpha = 0.05 (the open question) and
# alpha = 0.1 (one recipe for every system). C2: 4x data (96 1x-box trajectories), k_max 48 (k_mid 24), 12000
# propagator steps; C3: C2 + --periodic-full (whole 8-site-wide box supervised, reach 3 < half the box).
# Read-out (written before the runs, Block Z Z8): rollout statistics at 1x/2x/3x (early read: large-scale drift),
# then D_KY at 2x IC 1 (T120 truth) and 3x (IC 1; IC 0 at alpha = 0.05, its only 3x truth), then the other ICs for whichever passes.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; CR=$A/experiments/crop; D=$A/datasets
log() { echo "$(date '+%F %T') [R14] $*" >> $L/r4.log; }
for al in 0.05 0.1; do DS=$D/kolmogorov_box1x_large_a$al.h5
  [ -f $DS ] || { log "gen $DS"; $PY scripts/gen_kolmogorov_box_large.py --alpha $al --workers 6 --out $DS > $L/gen_kbox_$al.log 2>&1; log "gen rc=$? $(tail -1 $L/gen_kbox_$al.log)"; }
done
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
truth() { a=$1; sc=$2; s=$3; if [ $a = 0.05 ]; then [ $sc = 3 ] && echo $CR/truth_a005_x3_seed0.json || echo $CR/truth_a0p05_x${sc}_seed${s}_T120.json
  else echo $CR/truth_a0p1_x${sc}_seed${s}_T120.json; fi; }
ev() { tag=$1; a=$2; sc=$3; s=$4; out=$CR/${tag}_x${sc}_seed$s.json; [ -f $out ] && return
  $PY scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s --truth-json $(truth $a $sc $s) >> $L/crop_eval_$tag.log 2>&1
  log "eval $tag x$sc seed$s rc=$? $($PY scripts/r4_summary.py dky $out)"; }
waitslot() { while [ $(jobs -rp | wc -l) -ge 3 ]; do sleep 30; done; }
for al in 0.05 0.1; do for v in C2 C3; do
  tag=r14_K${v}_a${al}_seed0; extra=""; [ $v = C3 ] && extra="--periodic-full"
  if [ ! -f $A/stage2_prop_kolmogorov_crop_$tag.pt ]; then log "train $tag start"
    $PY scripts/train_kolmogorov_crop.py --tag $tag --dataset $D/kolmogorov_box1x_large_a$al.h5 --n-offsets 16 \
      --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 48 --k-mid 24 \
      --noise-end 0.02 --prop-steps 12000 --crop-sites 8 --window 3 --prop-layers 1 --local-channels 6 $extra \
      > $L/crop_train_$tag.log 2>&1
    log "train $tag rc=$? $(grep -E 'step 12000/12000' $L/crop_train_$tag.log | tail -1)"; fi
  waitslot; ( $PY scripts/exp_rollout_stats_kolmogorov.py --tag $tag --scales 1,2,3 --n-steps 1000 --with-truth > $L/r14_rollout_$tag.log 2>&1
    log "rollout $tag rc=$? $(grep '^\[rollout\]' $L/r14_rollout_$tag.log | sed -E 's/\{.*share_k_lt_2_late\": ([0-9.]+).*\| truth.*share_k_lt_2_late\": ([0-9.]+).*/k<2 model \1 truth \2/' | tr '\n' ' ')" ) &
  waitslot; ev $tag $al 2 1 &
  s3=1; [ $al = 0.05 ] && s3=0  # alpha = 0.05 has a 3x truth for IC 0 only
  waitslot; ev $tag $al 3 $s3 &
done; done
wait; log "R14 DONE"
