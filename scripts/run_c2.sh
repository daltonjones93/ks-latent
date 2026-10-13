#!/bin/bash
# R13 (user 2026-10-12): C seed 2 transfers badly at 66^2 (169.5 vs 192.5, -12%; seeds 0/1 -4.7/-4.6%). Earlier
# seed failures were fixed by a longer rollout and more data; try that, plus one structural change, on ALL seeds:
#   C2: 144 trajectories (ks2d_box44_large.h5, 4x C's data), k_max 48 (k_mid 24), 12000 propagator steps.
#   C3: C2 + --periodic-full: train on the whole periodic 8x8 box (reach 3 < half the box, no self-wrap), so every
#       site is supervised (64 per sample) instead of the 2x2 free interior of a forced-halo window.
# Seeds 0-2 each. Pre-registered read-out: 66^2 D_KY at IC 0 (truth 192.50) for all six models, then 44^2 IC 0 and
# 66^2 IC 1 (185.60) for the better recipe; R9a (every seed within 5%) decides. Both recipes and all seeds are
# reported whatever the outcome. Load-capped (machine runs hot): trainings serial on MPS, one eval at a time, 2 threads.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; CR=$A/experiments/crop
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
log() { echo "$(date '+%F %T') [R13] $*" >> $L/r4.log; }
ev() { tag=$1; sc=$2; s=$3; t=$4; out=$CR/${tag}_x${sc}_seed$s.json; [ -f $out ] && return
  $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s --true-dky $t >> $L/crop_eval_$tag.log 2>&1
  log "eval $tag x$sc seed$s rc=$? $($PY scripts/r4_summary.py dky $out)"; }
waitslot() { while [ $(jobs -rp | wc -l) -ge 1 ]; do sleep 30; done; }  # 21:26: one eval at a time (heat)
# a training left running by the previous instance of this script finishes first
while pgrep -f "train_kolmogorov_crop.py --tag r13_" > /dev/null; do sleep 60; done
for s in 0 1 2; do for v in C2 C3; do
  tag=r13_${v}_ks2d_s44big_k48_seed$s; extra=""; [ $v = C3 ] && extra="--periodic-full"
  if [ ! -f $A/stage2_prop_kolmogorov_crop_$tag.pt ]; then log "train $tag start"
    $PY scripts/train_kolmogorov_crop.py --tag $tag --seed $s --dataset $A/datasets/ks2d_box44_large.h5 --n-offsets 16 \
      --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 48 --k-mid 24 \
      --noise-end 0.02 --prop-steps 12000 --crop-sites 8 --crop-sites-y 8 --window 3 --prop-layers 1 --local-channels 6 \
      $extra > $L/crop_train_$tag.log 2>&1
    log "train $tag rc=$? $(grep -E 'step 12000/12000' $L/crop_train_$tag.log | tail -1)"; fi
  waitslot; ev $tag 1.5 0 192.50 &
done; done
wait; log "R13 66^2 IC0 evals done"
