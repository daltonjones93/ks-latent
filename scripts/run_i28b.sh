#!/bin/bash
# I28 stage 1b: extend both filters' tuning past the sweep edges (seed 0, stride 8).
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i28.log
lat() { tag=$1; shift
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag p7c12_ks2d --n-tilings 4 --out $tag "$@" > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep -E "^$tag |DIVERGED" artifacts/logs/h9_${tag}.log | cut -c1-200 | tr '\n' ' ')" >> $LOG; }
pde() { tag=$1; shift
  [ -f artifacts/experiments/da2d/i11_${tag}.json ] && return
  $PY scripts/exp_i11_pde_letkf.py --workers 4 --loc-c 16 --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/i11_${tag}.log | cut -c1-200)" >> $LOG; }
case $1 in
  lat) for A in 1.1 1.25; do
         lat I28_a${A}_lat_rtps1.0 --truth-lap-coef $A --rtps 1.0
         lat I28_a${A}_lat_rtps0.9_inf1.2 --truth-lap-coef $A --rtps 0.9 --inflation 1.2
         lat I28_a${A}_lat_rtps0.9_mn0.05 --truth-lap-coef $A --rtps 0.9 --model-noise-std 0.05
       done ;;
  pde) for A in 1.1 1.25; do for r in 0.8 0.9; do pde I28_a${A}_pde_rtps$r --truth-lap-coef $A --rtps $r; done; done ;;
esac
echo "$(date '+%F %T') LANE $1 (1b) DONE" >> $LOG
