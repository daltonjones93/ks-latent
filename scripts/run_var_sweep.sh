#!/bin/bash
# Accuracy study: variational recentring sweep at 44^2 (per-site filter, N=64, 100 cycles).
# Args: gamma beta... (one lane per gamma)
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
AE=artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt; PR=artifacts/stage2_prop_kolmogorov_crop_p7b_ks2d.pt
g=$1; shift
for b in "$@"; do
  tag=VAR_L44_mn0.02_g${g}_b${b}
  [ -f artifacts/experiments/da2d/da2d_${tag}.json ] && continue
  OMP_NUM_THREADS=3 $PY scripts/run_da_pff_2d_ltransfer.py --ae-checkpoint $AE --prop-checkpoint $PR --n-cycles 100 \
    --obs-stride 8 --L 44 --n-ensemble 64 --method letkf_local2d --model-noise-std 0.02 --phys-rmse \
    --var-recenter --var-gamma $g --var-beta $b --seed 0 --tag $tag > artifacts/logs/var_${tag}.log 2>&1
done
