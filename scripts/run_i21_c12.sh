#!/bin/bash
# I21 on the 12-channel model (Part II's model from 2026-10-09): inflation and RTPS, K = 4, 44^2, seed 0.
# Stride 8: 100 cycles (steady). Stride 16: 300 cycles (drift test: cycles 151-225 vs 226-300).
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i21_c12.log
run() { tag=$1; shift
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && return
  OMP_NUM_THREADS=4 $PY scripts/exp_h9_tiled_da.py --tag p7c12_ks2d --n-tilings 4 --out $tag "$@" > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/h9_${tag}.log)" >> $LOG; }
run L44_C12_I21_st16_base --obs-stride 16 --n-cycles 300
for r in 0.3 0.5 0.7; do run L44_C12_I21_st16_rtps$r --obs-stride 16 --n-cycles 300 --rtps $r; done
for f in 1.2 1.5; do run L44_C12_I21_st16_inf$f --obs-stride 16 --n-cycles 300 --inflation $f; done
for r in 0.3 0.5 0.7; do run L44_C12_I21_st8_rtps$r --obs-stride 8 --rtps $r; done
for f in 1.2 1.5; do run L44_C12_I21_st8_inf$f --obs-stride 8 --inflation $f; done
echo "$(date '+%F %T') I21 C12 DONE" >> $LOG
