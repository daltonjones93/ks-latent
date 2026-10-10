#!/bin/bash
# Part II required runs with the canonical Part II setup (2026-10-09): 12-channel p7c12_ks2d, Hann
# blend, RTPS 0.7, GC c = 1.5 sites, stride 8, 100 cycles. Sequential (large domains one at a time).
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/partII_c12.log
run() { tag=$1; shift
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && { echo "skip $tag" >> $LOG; return; }
  echo "$(date '+%F %T') start $tag" >> $LOG
  OMP_NUM_THREADS=6 /usr/bin/time -l $PY scripts/exp_h9_tiled_da.py --tag p7c12_ks2d --rtps 0.7 --out $tag "$@" \
    > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep -E "^$tag |DIVERGED" artifacts/logs/h9_${tag}.log | cut -c1-220 | tr '\n' ' ')" >> $LOG; }
# I2: equal total members (64), seeds 0-2
for sd in 0 1 2; do
  run P2_L44_I2_T1_N64_s$sd --n-tilings 1 --n-ensemble 64 --seed $sd
  run P2_L44_I2_T2_N32_s$sd --n-tilings 2 --n-ensemble 32 --seed $sd
  run P2_L44_I2_T4_N16_s$sd --n-tilings 4 --n-ensemble 16 --seed $sd
  run P2_L44_I2_T16_N4_s$sd --n-tilings 16 --n-ensemble 4 --seed $sd
done
# I4: number of tilings and weights (N = 64 per tiling), seed 0
run P2_L44_I4_T2 --n-tilings 2
run P2_L44_I4_T4 --n-tilings 4
run P2_L44_I4_T16 --n-tilings 16
run P2_L44_I4_T4u --n-tilings 4 --weights uniform
run P2_L44_I4_T16u --n-tilings 16 --weights uniform
# I1: scaling, K = 1 and 4, seed 0; then seed 1 at 220^2 and 330^2
for L in 88 220 330 440; do
  run P2_L${L}_I1_T4 --L $L --n-tilings 4
  run P2_L${L}_I1_T1 --L $L --n-tilings 1
done
run P2_L220_I1_T4_s1 --L 220 --n-tilings 4 --seed 1
run P2_L330_I1_T4_s1 --L 330 --n-tilings 4 --seed 1
echo "$(date '+%F %T') PARTII C12 QUEUE DONE" >> $LOG
