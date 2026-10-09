#!/bin/bash
# I12 pivot test (2026-10-09): 12-channel 2D KS model (p7c12_ks2d) vs the 6-channel p7b_ks2d.
# Waits for Stage B to finish, then: (1) blended DA at 44^2 (K = 1, 4, 16; strides 8, 16; seed 0;
# K = 4 seeds 1-2) -- minutes; (2) Lyapunov transfer at 44^2 and 66^2 (H1 rule i) -- ~1.5 h.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i12_pivot.log
PR=artifacts/stage2_prop_kolmogorov_crop_p7c12_ks2d.pt
until [ -f $PR ] && ! pgrep -f "train_kolmogorov_crop.py.*p7c12_ks2d" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') Stage B done; starting" >> $LOG
run() { tag=$1; shift
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && return
  OMP_NUM_THREADS=6 $PY scripts/exp_h9_tiled_da.py --tag p7c12_ks2d --out $tag "$@" > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/h9_${tag}.log)" >> $LOG; }
run L44_C12_T1_st8 --n-tilings 1
run L44_C12_T4_st8 --n-tilings 4
run L44_C12_T16_st8 --n-tilings 16
run L44_C12_T1_st16 --n-tilings 1 --obs-stride 16
run L44_C12_T4_st16 --n-tilings 4 --obs-stride 16
run L44_C12_T16_st16 --n-tilings 16 --obs-stride 16
run L44_C12_T4_st8_s1 --n-tilings 4 --seed 1
run L44_C12_T4_st8_s2 --n-tilings 4 --seed 2
echo "$(date '+%F %T') DA DONE" >> $LOG
for sc in 1.0 1.5; do
  nd=$([ "$sc" = "1.0" ] && echo 384 || echo 512)
  echo "$(date '+%F %T') start lyap scale $sc" >> $LOG
  OMP_NUM_THREADS=6 $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt artifacts/stage1_ae_kolmogorov_crop_p7c12_ks2d.pt \
    --prop-ckpt $PR --scale $sc --seed 0 --n-directions $nd --tag p7c12_ks2d_x${sc}_seed0 >> artifacts/logs/crop_eval_p7c12_ks2d.log 2>&1
  echo "$(date '+%F %T') done lyap scale $sc rc=$? $(grep -E 'rel_err_pct|model_D_KY|true_D_KY' artifacts/logs/crop_eval_p7c12_ks2d.log | tail -3 | tr -d '\n ')" >> $LOG
done
echo "$(date '+%F %T') I12 PIVOT DONE" >> $LOG
