#!/bin/bash
# J1/J2 follow-up (2026-10-09): cases where c = 8.25 beat c = 16 on seed 0 (mean flow 1/2, forcing 0.4).
# Seeds 1-2 at c = 8.25; c = 4 for the mean-flow cases (8.25 is the grid edge); J16(a) mean-flow parameter
# estimation repeated at the better c. Runs after J16 (lowest priority).
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/j1b.log
until grep -q "J16 DONE" artifacts/logs/j16.log 2>/dev/null; do sleep 120; done
ENV="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2"
pde() { tag=$1; shift
  [ -f artifacts/experiments/da2d/i11_${tag}.json ] && return
  env $ENV nice -n 19 $PY scripts/exp_i11_pde_letkf.py --workers 2 --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/i11_${tag}.log | cut -c1-170)" >> $LOG; }
for sd in 1 2; do
  pde J1b_adv1_c8.25_rtps0.9_s$sd --model-advect-x 1 --loc-c 8.25 --rtps 0.9 --seed $sd
  pde J1b_adv2_c8.25_rtps1.0_s$sd --model-advect-x 2 --loc-c 8.25 --rtps 1.0 --seed $sd
  pde J1b_frc0.4_c8.25_rtps0.7_s$sd --model-forcing-amp 0.4 --loc-c 8.25 --rtps 0.7 --seed $sd
done
pde J1b_adv1_c4_rtps0.9_s0 --model-advect-x 1 --loc-c 4 --rtps 0.9
pde J1b_adv2_c4_rtps1.0_s0 --model-advect-x 2 --loc-c 4 --rtps 1.0
for sd in 0 1 2; do
  pde J16a_adv1_c8.25_rtps0.9_s$sd --model-advect-x 1 --estimate-param adv --param-init-std 0.3 --loc-c 8.25 --rtps 0.9 --seed $sd
  pde J16a_adv2_c8.25_rtps1.0_s$sd --model-advect-x 2 --estimate-param adv --param-init-std 0.3 --loc-c 8.25 --rtps 1.0 --seed $sd
done
echo "$(date '+%F %T') J1b DONE" >> $LOG
