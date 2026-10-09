#!/bin/bash
# Accuracy study batch 2 (44^2, per-site, N=64, stride 8, mn 0.02, 100 cycles). Args: tag then extra flags.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
AE=artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt; PR=artifacts/stage2_prop_kolmogorov_crop_p7b_ks2d.pt
tag=$1; shift
[ -f artifacts/experiments/da2d/da2d_${tag}.json ] && exit 0
OMP_NUM_THREADS=2 $PY scripts/run_da_pff_2d_ltransfer.py --ae-checkpoint $AE --prop-checkpoint $PR --n-cycles 100 \
  --obs-stride 8 --L 44 --n-ensemble 64 --method letkf_local2d --model-noise-std 0.02 --phys-rmse --seed 0 \
  --tag $tag "$@" > artifacts/logs/acc2_${tag}.log 2>&1
