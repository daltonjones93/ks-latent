#!/bin/bash
# I31 follow-up: 600-cycle runs (gate orig/ft and projection, seeds 0-2, stride 16) and the gate at stride 8.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i31_long.log
run() { tag=$1; shift
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag p7c12_ks2d --n-tilings 4 --out $tag "$@" > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep -E "^$tag |DIVERGED" artifacts/logs/h9_${tag}.log | cut -c1-160 | tr '\n' ' ')" >> $LOG; }
case $1 in
  g) for sd in 0 1 2; do run L44_C12G_ft_st16_600_s$sd --gate --gate-off-tag p7c12n15_ks2d --obs-stride 16 --n-cycles 600 --rtps 0.5 --seed $sd; done ;;
  p) for sd in 0 1 2; do run L44_C12P_st16_600_s$sd --project-members --obs-stride 16 --n-cycles 600 --rtps 0.5 --seed $sd; done ;;
  d) run L44_C12G_ft_st8_rtps0.7 --gate --gate-off-tag p7c12n15_ks2d --obs-stride 8 --rtps 0.7
     run L44_C12G_ft_st8_rtps0.7_s1 --gate --gate-off-tag p7c12n15_ks2d --obs-stride 8 --rtps 0.7 --seed 1
     run L44_C12_st8_rtps0.7_s1 --obs-stride 8 --rtps 0.7 --seed 1 ;;
esac
echo "$(date '+%F %T') LANE $1 DONE" >> $LOG
