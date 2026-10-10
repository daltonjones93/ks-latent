#!/bin/bash
# I28, user design (2026-10-09): truth = the standard 2D KS (the system the latent model was trained
# on); the physics filter's MODEL is wrong. The latent filter's numbers are the standard-truth runs.
# Seed 0, stride 8, 100 cycles, PDE filter c = 16 units.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i28d.log
pde() { tag=$1; shift
  [ -f artifacts/experiments/da2d/i11_${tag}.json ] && return
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
    $PY scripts/exp_i11_pde_letkf.py --workers 2 --loc-c 16 --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/i11_${tag}.log | cut -c1-200)" >> $LOG; }
R="0.7 0.5 0.9 0"
case $1 in
  lap) for v in 1.1 1.25; do for r in $R; do pde I28W_lap${v}_rtps$r --model-lap-coef $v --rtps $r; done; done
       for M in 32 64; do for r in 0.7 0.5 0; do pde I28R_M${M}_pde_rtps$r --model-nx $M --rtps $r; done; done ;;
  adv) for v in 1 2; do for r in $R; do pde I28W_adv${v}_rtps$r --model-advect-x $v --rtps $r; done; done ;;
  frc) for v in 0.2 0.4; do for r in $R; do pde I28W_frc${v}_rtps$r --model-forcing-amp $v --rtps $r; done; done ;;
  res) for M in 32 64; do for r in 0.5 0.7 0; do pde I28R_M${M}_pde_rtps$r --model-nx $M --rtps $r; done; done
       [ -f artifacts/experiments/da2d/h9_L44_C12_st8_rtps0.7_s2.json ] || \
         OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag p7c12_ks2d --n-tilings 4 --obs-stride 8 --rtps 0.7 --seed 2 \
           --out L44_C12_st8_rtps0.7_s2 > artifacts/logs/h9_L44_C12_st8_rtps0.7_s2.log 2>&1
       echo "$(date '+%F %T') done latent seed2 $(grep '^L44' artifacts/logs/h9_L44_C12_st8_rtps0.7_s2.log | cut -c1-120)" >> $LOG ;;
esac
echo "$(date '+%F %T') LANE $1 DONE" >> $LOG
