#!/bin/bash
# I11 sweep extension (2026-10-09 08:03): the best localization half-width so far is the largest
# in the declared grid (c = 12 units), so the grid is extended upward. Runs after run_i11_sweep.sh.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i11_sweep.log
until grep -q "I11 SWEEP DONE" $LOG 2>/dev/null; do sleep 60; done
run() { tag=$1; shift
  [ -f artifacts/experiments/da2d/i11_${tag}.json ] && return
  echo "$(date '+%F %T') start $tag" >> $LOG
  $PY scripts/exp_i11_pde_letkf.py --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/i11_${tag}.log)" >> $LOG; }
for st in 8 16; do
  run L44_st${st}_c16_rtps0 --obs-stride $st --loc-c 16 --rtps 0
  run L44_st${st}_c24_rtps0 --obs-stride $st --loc-c 24 --rtps 0
  run L44_st${st}_noloc_rtps0 --obs-stride $st --loc-c 0 --rtps 0
done
echo "$(date '+%F %T') I11 SWEEP2 DONE" >> $LOG
