#!/bin/bash
# I33: evaluate the 12-channel model trained on noisy data (p7c12d10_ks2d, data noise 0.1) once Stage B
# exists. Scored against the CLEAN truth. Same Part II settings as the clean model's runs.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i33.log
TAG=p7c12d10_ks2d
PR=artifacts/stage2_prop_kolmogorov_crop_$TAG.pt
until [ -f $PR ] && ! pgrep -f "train_kolmogorov_crop.py.*$TAG" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') training done; evaluating" >> $LOG
run() { out=$1; shift
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag $TAG --obs-stride 8 --rtps 0.7 --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
  echo "$(date '+%F %T') done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-200 | tr '\n' ' ')" >> $LOG; }
for sd in 0 1 2; do run I33_d10_T4_s$sd --n-tilings 4 --seed $sd; done
run I33_d10_T1_s0 --n-tilings 1
OMP_NUM_THREADS=4 $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt artifacts/stage1_ae_kolmogorov_crop_$TAG.pt --prop-ckpt $PR \
  --scale 1.0 --seed 0 --n-directions 384 --tag ${TAG}_x1.0_seed0 > artifacts/logs/crop_eval_$TAG.log 2>&1
echo "$(date '+%F %T') done lyap $(grep -E 'rel_err_pct|model_D_KY' artifacts/logs/crop_eval_$TAG.log | tr -d '\n ')" >> $LOG
echo "$(date '+%F %T') I33 EVAL DONE" >> $LOG
