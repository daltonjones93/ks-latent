#!/bin/bash
# I20 (cycle length) and I21 (wider localization) on the 6-channel model, K = 4, 44^2, seed 0.
# Total assimilation window fixed at 50 time units (n_cycles = 500 / n_prop_steps); steady = last half.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i20_i21.log
run() { tag=$1; shift
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && return
  OMP_NUM_THREADS=4 $PY scripts/exp_h9_tiled_da.py --out $tag "$@" > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/h9_${tag}.log)" >> $LOG; }
for st in 8 16; do
  for c in 2 3 4; do run L44_I21_T4_st${st}_c$c --n-tilings 4 --obs-stride $st --gc-c $c; done
  run L44_I20_T4_st${st}_np1 --n-tilings 4 --obs-stride $st --n-prop-steps 1 --n-cycles 500
  run L44_I20_T4_st${st}_np2 --n-tilings 4 --obs-stride $st --n-prop-steps 2 --n-cycles 250
  run L44_I20_T4_st${st}_np10 --n-tilings 4 --obs-stride $st --n-prop-steps 10 --n-cycles 50
done
echo "$(date '+%F %T') I20/I21 DONE" >> $LOG
