#!/bin/bash
# Task I11 sweep: perfect-model local ETKF at 44^2, strides 8 and 16. Ceiling = best steady RMSE
# over this declared grid (localization half-width c in physical units x RTPS), plus no localization.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i11_sweep.log
run() { tag=$1; shift
  [ -f artifacts/experiments/da2d/i11_${tag}.json ] && return
  echo "$(date '+%F %T') start $tag" >> $LOG
  $PY scripts/exp_i11_pde_letkf.py --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/i11_${tag}.log)" >> $LOG; }
for st in 8 16; do
  for c in 4 8.25 12; do
    for r in 0 0.5 0.9; do
      run L44_st${st}_c${c}_rtps${r} --obs-stride $st --loc-c $c --rtps $r
    done
  done
  run L44_st${st}_noloc_rtps0.5 --obs-stride $st --loc-c 0 --rtps 0.5
done
echo "$(date '+%F %T') I11 SWEEP DONE" >> $LOG
