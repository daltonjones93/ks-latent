#!/bin/bash
# Part II reruns with Part II's model from 2026-10-09: p7c12d10_ks2d (12 channels, trained only on data with
# observation-level noise). Settings as before: K = 4, Hann, RTPS 0.7, GC c = 1.5 sites, stride 8, 100 cycles.
# Lane a: 44^2 work (OMP 3). Lane b: sizes 88^2-440^2, one at a time (OMP 6).
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
TAG=p7c12d10_ks2d
LOG=artifacts/logs/partII_d10.log
run() { th=$1; out=$2; shift 2
  [ -f artifacts/experiments/da2d/h9_${out}.json ] && { echo "skip $out" >> $LOG; return; }
  echo "$(date '+%F %T') start $out" >> $LOG
  OMP_NUM_THREADS=$th /usr/bin/time -l $PY scripts/exp_h9_tiled_da.py --tag $TAG --rtps 0.7 --out $out "$@" > artifacts/logs/h9_${out}.log 2>&1
  echo "$(date '+%F %T') done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_${out}.log | cut -c1-200 | tr '\n' ' ')" >> $LOG; }
emu() { out=$1; shift
  [ -f artifacts/experiments/da2d/i24_${out}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_i24_phys_emulator.py --tag $TAG --out $out "$@" > artifacts/logs/i24_${out}.log 2>&1
  echo "$(date '+%F %T') done $out rc=$? $(grep "^$out " artifacts/logs/i24_${out}.log | cut -c1-160)" >> $LOG; }
case $1 in
 a)
  # RTPS check (pre-registered default stays 0.7 unless another value is >5% better AND calibrated)
  run 3 D10_L44_rtps0.5 --n-tilings 4 --rtps 0.5
  run 3 D10_L44_rtps0.9 --n-tilings 4 --rtps 0.9
  # I2 equal total members (64), seeds 0-2 (4 x 64 per tiling = I33_d10_T4_s*)
  for sd in 0 1 2; do
    run 3 D10_L44_I2_T1_N64_s$sd --n-tilings 1 --n-ensemble 64 --seed $sd
    run 3 D10_L44_I2_T2_N32_s$sd --n-tilings 2 --n-ensemble 32 --seed $sd
    run 3 D10_L44_I2_T4_N16_s$sd --n-tilings 4 --n-ensemble 16 --seed $sd
  done
  # I4 number of tilings and weights
  run 3 D10_L44_I4_T2 --n-tilings 2
  run 3 D10_L44_I4_T16 --n-tilings 16
  run 3 D10_L44_I4_T4u --n-tilings 4 --weights uniform
  # I3 ensemble size per tiling (N = 64 exists)
  for N in 8 16 32 128; do run 3 D10_L44_I3_N$N --n-tilings 4 --n-ensemble $N; done
  # I8 / I9 / I10 diagnostics and fields
  run 3 D10_L44_diag_T4 --n-tilings 4 --diag --save-fields 10
  run 3 D10_L44_fields_T1 --n-tilings 1 --save-fields 10
  # I24 physical-space analysis with the same (noisy-data) model
  for sd in 0 1 2; do emu D10_I24_T4_c8.25_rtps0.5_s$sd --n-tilings 4 --loc-c 8.25 --rtps 0.5 --seed $sd; done
  emu D10_I24_T1_c8.25_rtps0.5 --n-tilings 1 --loc-c 8.25 --rtps 0.5
  # Lyapunov at 66^2 for Part II's model
  OMP_NUM_THREADS=3 $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt artifacts/stage1_ae_kolmogorov_crop_$TAG.pt \
    --prop-ckpt artifacts/stage2_prop_kolmogorov_crop_$TAG.pt --scale 1.5 --seed 0 --n-directions 512 \
    --tag ${TAG}_x1.5_seed0 > artifacts/logs/crop_eval_${TAG}_x1.5.log 2>&1
  echo "$(date '+%F %T') done lyap66 $(grep -E 'rel_err_pct|model_D_KY' artifacts/logs/crop_eval_${TAG}_x1.5.log | tr -d '\n ')" >> $LOG
  $PY scripts/exp_i9_seams.py artifacts/experiments/da2d/h9_D10_L44_diag_T4_fields.npz artifacts/experiments/da2d/h9_D10_L44_fields_T1_fields.npz >> $LOG 2>&1
  ;;
 b)
  for L in 88 220 330 440; do
    run 6 D10_L${L}_I1_T4 --L $L --n-tilings 4
    run 6 D10_L${L}_I1_T1 --L $L --n-tilings 1
  done
  run 6 D10_L220_I1_T4_s1 --L 220 --n-tilings 4 --seed 1
  run 6 D10_L330_I1_T4_s1 --L 330 --n-tilings 4 --seed 1
  run 6 D10_L220_diag_T4 --L 220 --n-tilings 4 --diag --save-fields 6
  for N in 16 32 128; do run 6 D10_L220_I3_N$N --L 220 --n-tilings 4 --n-ensemble $N; done
  ;;
esac
echo "$(date '+%F %T') LANE $1 DONE" >> $LOG
