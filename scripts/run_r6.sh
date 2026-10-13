#!/bin/bash
# R6 (2026-10-12): matched ensemble budget for the J17G missing-process comparison. Paper II P7 compares the latent
# filter (4 tilings x 64 members) with the physics filter (64 members); here the latent filter runs with 4 x 16 = 64
# member-states in total, same tag, RTPS, observations and seeds as run_j17g.sh's latent runs.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
E=artifacts/experiments/da2d; LOG=artifacts/logs/j17g.log
for d in 0.1 0.2 0.3; do TAG=p7c12d10_gani$d; P=J17G_d$d
  for s in 0 1 2; do out=${P}_lat_T4_N16_s$s; [ -f $E/h9_$out.json ] && continue
    OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag $TAG --obs-stride 8 --rtps 0.7 --truth-ganiso $d --n-tilings 4 \
      --n-ensemble 16 --seed $s --out $out > artifacts/logs/h9_$out.log 2>&1
    echo "$(date '+%F %T') [R6] $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-200)" >> $LOG; done; done
echo "$(date '+%F %T') [R6] DONE" >> $LOG
