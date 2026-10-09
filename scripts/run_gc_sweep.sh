#!/bin/bash
# Localization radius x observation density at 44^2 (per-site filter, N=64, model noise 0.02, 100 cycles).
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
AE=artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt; PR=artifacts/stage2_prop_kolmogorov_crop_p7b_ks2d.pt
st=$1; shift
for c in "$@"; do
  tag=GC_L44_st${st}_c${c}_mn0.02
  [ -f artifacts/experiments/da2d/da2d_${tag}.json ] && continue
  OMP_NUM_THREADS=2 $PY scripts/run_da_pff_2d_ltransfer.py --ae-checkpoint $AE --prop-checkpoint $PR --n-cycles 100 \
    --obs-stride $st --L 44 --n-ensemble 64 --method letkf_local2d --model-noise-std 0.02 --gc-c $c --phys-rmse \
    --seed 0 --tag $tag > artifacts/logs/gc_${tag}.log 2>&1
done
