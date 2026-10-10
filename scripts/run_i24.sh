#!/bin/bash
# I24 equation-free physical-space baseline sweep (seed 0, stride 8, 12-channel model).
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i24.log
run() { tag=$1; shift
  [ -f artifacts/experiments/da2d/i24_${tag}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_i24_phys_emulator.py --out $tag "$@" > artifacts/logs/i24_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep -E "^$tag |DIVERGED" artifacts/logs/i24_${tag}.log | cut -c1-200 | tr '\n' ' ')" >> $LOG; }
for K in 4 1; do for c in 8.25 16; do for r in 0 0.5 0.7; do
  run I24_T${K}_c${c}_rtps$r --n-tilings $K --loc-c $c --rtps $r
done; done; done
echo "$(date '+%F %T') I24 SWEEP DONE" >> $LOG
