#!/bin/bash
# I30(d) follow-up: seeds 1-2 for both propagators (stride 16, RTPS 0.5, 300 cycles) and the
# member-projection baseline (z <- E(D(z)) after each forecast; original propagator), seeds 0-2.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i30_seeds_proj.log
run() { tag=$1; model=$2; shift 2
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && return
  OMP_NUM_THREADS=4 $PY scripts/exp_h9_tiled_da.py --tag $model --n-tilings 4 --obs-stride 16 --n-cycles 300 \
    --out $tag "$@" > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/h9_${tag}.log | cut -c1-200)" >> $LOG; }
lane=$1
if [ "$lane" = a ]; then
  for sd in 1 2; do
    run L44_C12_st16_rtps0.5_s$sd p7c12_ks2d --rtps 0.5 --seed $sd
    run L44_C12N15_st16_rtps0.5_s$sd p7c12n15_ks2d --rtps 0.5 --seed $sd
  done
else
  for sd in 0 1 2; do run L44_C12P_st16_rtps0.5_s$sd p7c12_ks2d --rtps 0.5 --project-members --seed $sd; done
  run L44_C12P_st16_base_s0 p7c12_ks2d --project-members --seed 0
fi
echo "$(date '+%F %T') LANE $lane DONE" >> $LOG
