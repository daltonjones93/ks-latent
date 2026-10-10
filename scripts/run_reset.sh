#!/bin/bash
# Tiling-reset failsafe test (2026-10-09): stride 16, RTPS 0.5, 600 cycles, seeds 0-2.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/reset.log
run() { tag=$1; shift
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag p7c12_ks2d --n-tilings 4 --tiling-reset --out $tag "$@" > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep -E "^$tag |DIVERGED" artifacts/logs/h9_${tag}.log | cut -c1-170 | tr '\n' ' ')" >> $LOG; }
case $1 in
  g) for sd in 0 1 2; do run L44_C12GR_st16_600_s$sd --gate --gate-off-tag p7c12n15_ks2d --obs-stride 16 --n-cycles 600 --rtps 0.5 --seed $sd; done ;;
  o) for sd in 0 1 2; do run L44_C12R_st16_600_s$sd --obs-stride 16 --n-cycles 600 --rtps 0.5 --seed $sd; done
     run L44_C12GR_st8 --gate --gate-off-tag p7c12n15_ks2d --obs-stride 8 --rtps 0.7 ;;
esac
echo "$(date '+%F %T') LANE $1 DONE" >> $LOG
