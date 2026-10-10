#!/bin/bash
# I28 imperfect-model twin (2026-10-09): truth with coefficient a on nabla^2; both filters use a = 1.
# Stage 1 (seed 0): tune each filter's RTPS. Stage 2 (seeds 1-2) is launched after reading stage 1.
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
  lat) for A in 1.1 1.25; do for r in 0.5 0.7 0.9; do lat I28_a${A}_lat_rtps$r --truth-lap-coef $A --rtps $r; done; done ;;
  pde) for A in 1.1 1.25; do for r in 0 0.3 0.5 0.7; do pde I28_a${A}_pde_rtps$r --truth-lap-coef $A --rtps $r; done; done ;;
esac
echo "$(date '+%F %T') LANE $1 DONE" >> $LOG
