#!/bin/bash
# I31 + I30(d) combination test: 12-channel, K = 4, stride 16, RTPS 0.5, 300 cycles, seeds 0-2.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i31_gate.log
run() { tag=$1; model=$2; shift 2
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && return
  OMP_NUM_THREADS=4 $PY scripts/exp_h9_tiled_da.py --tag $model --n-tilings 4 --obs-stride 16 --n-cycles 300 \
    --rtps 0.5 --out $tag "$@" > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep -E "^$tag |DIVERGED" artifacts/logs/h9_${tag}.log | cut -c1-160 | tr '\n' ' ')" >> $LOG; }
case $1 in
  a) for sd in 0 1 2; do run L44_C12N15P_st16_s$sd p7c12n15_ks2d --project-members --seed $sd; done ;;
  b) for sd in 0 1 2; do run L44_C12G_proj_st16_s$sd p7c12_ks2d --gate --seed $sd; done ;;
  c) for sd in 0 1 2; do run L44_C12G_ft_st16_s$sd p7c12_ks2d --gate --gate-off-tag p7c12n15_ks2d --seed $sd; done ;;
esac
echo "$(date '+%F %T') LANE $1 DONE" >> $LOG
