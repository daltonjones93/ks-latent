#!/bin/bash
# R13 follow-up (pre-registered read-out, Block Z R13): C2 is the better recipe on 66^2 IC 0 (all seeds within 5.2% of
# IC 0's truth 192.50, within 3.4% of the truth mean 189.05; C3: up to 6.5%). Now 44^2 IC 0 (81.77) and 66^2 IC 1
# (185.60) for all C2 seeds, 88^2 IC 0 for all C2 seeds (truth pending; extensive prediction ~333), and the remaining
# ICs for C2 seed 0 (44^2 IC 1-2: 80.66, 83.60; 66^2 IC 2 against its truth when it lands). Two evals at a time, 3 threads.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; L=$A/logs; CR=$A/experiments/crop
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
log() { echo "$(date '+%F %T') [R13b] $*" >> $L/r4.log; }
ev() { tag=$1; sc=$2; s=$3; t=$4; out=$CR/${tag}_x${sc}_seed$s.json; [ -f $out ] && return
  extra=""; [ "$t" != "-" ] && extra="--true-dky $t"; [ $sc = 2.0 ] && extra="$extra --n-directions 512"
  $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s $extra >> $L/crop_eval_$tag.log 2>&1
  log "eval $tag x$sc seed$s rc=$? $($PY scripts/r4_summary.py dky $out)"; }
waitslot() { while [ $(jobs -rp | wc -l) -ge 2 ]; do sleep 30; done; }
T=r13_C2_ks2d_s44big_k48_seed
for s in 0 1 2; do waitslot; ev $T$s 1.0 0 81.77 & done
for s in 0 1 2; do waitslot; ev $T$s 1.5 1 185.60 & done
for s in 0 1 2; do waitslot; ev $T$s 2.0 0 - & done
waitslot; ev ${T}0 1.0 1 80.66 & waitslot; ev ${T}0 1.0 2 83.60 &
wait; log "R13b DONE"
