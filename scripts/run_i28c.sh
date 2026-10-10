#!/bin/bash
# I28 options 1 (resolution error, PDE filter on a coarse grid; standard truth) and 2 (missing
# processes: mean-flow advection or terrain forcing in the truth; neither filter has them).
# Seed 0, stride 8, 100 cycles.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i28c.log
lat() { tag=$1; shift
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag p7c12_ks2d --n-tilings 4 --out $tag "$@" > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep -E "^$tag |DIVERGED" artifacts/logs/h9_${tag}.log | cut -c1-200 | tr '\n' ' ')" >> $LOG; }
pde() { tag=$1; shift
  [ -f artifacts/experiments/da2d/i11_${tag}.json ] && return
  $PY scripts/exp_i11_pde_letkf.py --workers 3 --loc-c 16 --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/i11_${tag}.log | cut -c1-200)" >> $LOG; }
TRUTHS=("adv1:--truth-advect-x 1.0" "adv2:--truth-advect-x 2.0" "frc0.2:--truth-forcing-amp 0.2" "frc0.4:--truth-forcing-amp 0.4")
case $1 in
  a) for M in 32 64; do for r in 0 0.5 0.7; do pde I28R_M${M}_pde_rtps$r --model-nx $M --rtps $r; done; done
     for T in "${TRUTHS[@]}"; do name=${T%%:*}; args=${T#*:}
       lat I28S_${name}_lat_rtps0.7 $args --rtps 0.7
       lat I28S_${name}_lat_rtps0.9_mn0.05 $args --rtps 0.9 --model-noise-std 0.05
     done ;;
  b) for T in "${TRUTHS[@]}"; do name=${T%%:*}; args=${T#*:}
       for r in 0.5 0.7 0.9; do pde I28S_${name}_pde_rtps$r $args --rtps $r; done
     done ;;
esac
echo "$(date '+%F %T') LANE $1 DONE" >> $LOG
