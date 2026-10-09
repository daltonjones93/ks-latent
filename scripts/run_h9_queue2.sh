#!/bin/bash
# Block I follow-on queue (I2, I4, I8, I9). Waits for run_h9_queue.sh l220 to finish, then runs the
# fast 44^2 runs, then the 220^2 runs, one at a time. Never edit this file while it runs.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/h9_queue.log
until grep -q "QUEUE l220 DONE" $LOG 2>/dev/null; do sleep 60; done
run() { tag=$1; shift
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && { echo "skip $tag" >> $LOG; return; }
  echo "$(date '+%F %T') start $tag $*" >> $LOG
  OMP_NUM_THREADS=6 /usr/bin/time -l $PY scripts/exp_h9_tiled_da.py --out $tag "$@" \
    > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/h9_${tag}.log)" >> $LOG; }
# I8 + I9 at 44^2 (diagnostics, fields), strides 8 and 16
run L44_I89_T4 --n-tilings 4 --diag --save-fields 10
run L44_I89_T4_st16 --n-tilings 4 --obs-stride 16 --diag --save-fields 10
run L44_I89_T1 --n-tilings 1 --save-fields 10
# I2 equal total members (64), seeds 0-2
for sd in 0 1 2; do
  run L44_I2_T1_N64_s$sd --n-tilings 1 --n-ensemble 64 --seed $sd
  run L44_I2_T2_N32_s$sd --n-tilings 2 --n-ensemble 32 --seed $sd
  run L44_I2_T4_N16_s$sd --n-tilings 4 --n-ensemble 16 --seed $sd
done
# I4 number of tilings and weights, strides 8 and 16 (K=1 and Hann K=4 at stride 8 come from I2 / I89)
for st in 8 16; do
  run L44_I4_T2_st$st --n-tilings 2 --obs-stride $st
  run L44_I4_T16_st$st --n-tilings 16 --obs-stride $st
  run L44_I4_T4u_st$st --n-tilings 4 --weights uniform --obs-stride $st
done
run L44_I4_T1_st16 --n-tilings 1 --obs-stride 16
$PY scripts/exp_i9_seams.py artifacts/experiments/da2d/h9_L44_I89_*_fields.npz >> $LOG 2>&1
echo "$(date '+%F %T') QUEUE 44 DONE" >> $LOG
# 220^2: I2 equal-cost and I8/I9 diagnostics (K=1 N=64 baseline comes from the l220 queue)
run L220_I2_T4_N16 --L 220 --n-tilings 4 --n-ensemble 16
run L220_I2_T2_N32 --L 220 --n-tilings 2 --n-ensemble 32
run L220_I89_T4 --L 220 --n-tilings 4 --diag --save-fields 6
$PY scripts/exp_i9_seams.py artifacts/experiments/da2d/h9_L220_I89_T4_fields.npz >> $LOG 2>&1
echo "$(date '+%F %T') QUEUE2 DONE" >> $LOG
