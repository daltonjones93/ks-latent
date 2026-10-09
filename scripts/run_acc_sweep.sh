#!/bin/bash
# Accuracy study (2026-10-08): can the 2D latent filter match observation interpolation?
# 44^2, per-site filter, N = 64, 100 cycles, physical RMSE recorded. Lane: one model-noise value.
cd "$(dirname "$0")/.."
OUT=artifacts/experiments/da2d; LOG=artifacts/logs/acc_sweep.log
AE=artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt; PR=artifacts/stage2_prop_kolmogorov_crop_p7b_ks2d.pt
mn=$1; shift
for r in "$@"; do
  tag=ACC_L44_local_N64_mn${mn}_r${r}_s0
  [ -f "$OUT/da2d_${tag}.json" ] && continue
  echo "[acc $(date '+%H:%M:%S')] running $tag" >> $LOG
  OMP_NUM_THREADS=3 /opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python scripts/run_da_pff_2d_ltransfer.py \
    --ae-checkpoint $AE --prop-checkpoint $PR --n-cycles 100 --obs-stride 8 --L 44 --n-ensemble 64 \
    --method letkf_local2d --model-noise-std $mn --filter-obs-std $r --phys-rmse --seed 0 --tag $tag \
    > "$OUT/da2d_${tag}.stdout" 2>&1 && echo "[acc $(date '+%H:%M:%S')] done $tag" >> $LOG || echo "[acc] FAILED $tag" >> $LOG
done
