#!/bin/bash
# K1 (user, 2026-10-10): rerun the unlocalized-filter failure with the Part II model (p7c12d10_ks2d).
# Unlocalized = the same per-site ETKF with GC half-width 1e6 sites (taper = 1 to within 1e-9 everywhere,
# so every site uses every observation with the same weights: the global ETKF). 44^2 / 66^2 / 88^2,
# N = 64 and 128, K = 1 and K = 4, seed 0. Truth n+ = 42-44 (44^2), 93-101 (66^2), ~170 (88^2, model).
# Runs after K10(a) so the machine is not oversubscribed.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/k1_unloc.log; E=artifacts/experiments/da2d; TAG=p7c12d10_ks2d
until grep -q "(a) DONE" artifacts/logs/k10.log 2>/dev/null; do sleep 300; done
lat() { out=$1; shift; [ -f $E/h9_${out}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag $TAG --rtps 0.7 --obs-stride 8 --gc-c 1e6 --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
  echo "$(date '+%F %T') done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-200 | tr '\n' ' ')" >> $LOG; }
( for L in 44 66 88; do lat K1u_L${L}_T1_N64 --L $L --n-tilings 1 --n-ensemble 64; lat K1u_L${L}_T1_N128 --L $L --n-tilings 1 --n-ensemble 128; done ) &
( for L in 44 66 88; do lat K1u_L${L}_T4_N64 --L $L --n-tilings 4 --n-ensemble 64; lat K1u_L${L}_T4_N128 --L $L --n-tilings 4 --n-ensemble 128; done ) &
( lat K1l_L66_T4_N64 --L 66 --n-tilings 4 --n-ensemble 64; lat K1l_L66_T1_N64 --L 66 --n-tilings 1 --n-ensemble 64 ) &
wait
echo "$(date '+%F %T') K1 UNLOC DONE" >> $LOG
