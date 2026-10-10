#!/bin/bash
# J18a (2026-10-10): sigma = 0.3 noisy-data model. Waits for J17B's two trainings (one MPS training at a time),
# trains p7c12d30_ks2d (same recipe as p7c12d10_ks2d, --data-noise 0.3), then DA (44^2 K=4 seeds 0-2, K=1;
# 220^2 K=4 seed 0) and the J18b round-trip check.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/j18a.log; TAG=p7c12d30_ks2d; E=artifacts/experiments/da2d
log() { echo "$(date '+%F %T') $*" >> $LOG; }
until [ -f artifacts/stage2_prop_kolmogorov_crop_p7c12d10_memB0.2.pt ] && ! pgrep -f "train_kolmogorov_crop.py" > /dev/null; do sleep 300; done
if [ ! -f artifacts/stage2_prop_kolmogorov_crop_$TAG.pt ]; then
  log "training started"
  $PY scripts/train_kolmogorov_crop.py --dataset artifacts/datasets/ks2d_wide6.h5 --tag $TAG --crop-sites 40 --crop-sites-y 40 \
    --n-offsets 16 --local-channels 12 --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 \
    --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000 --data-noise 0.3 > artifacts/logs/crop_train_$TAG.log 2>&1
  log "training done rc=$?"
fi
run() { out=$1; shift
  [ -f $E/h9_${out}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag $TAG --obs-stride 8 --rtps 0.7 --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
  log "done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-200 | tr '\n' ' ')"; }
run J18a_d30_T4_s0 --n-tilings 4 --seed 0 & run J18a_d30_T4_s1 --n-tilings 4 --seed 1 & run J18a_d30_T4_s2 --n-tilings 4 --seed 2 & wait
run J18a_d30_T1_s0 --n-tilings 1 --save-fields 10 &
OMP_NUM_THREADS=2 $PY scripts/exp_j18b_roundtrip.py p7c12_ks2d p7c12d10_ks2d $TAG >> $LOG 2>&1
wait
run J18a_d30_L220_T4_s0 --n-tilings 4 --seed 0 --L 220
OMP_NUM_THREADS=4 $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt artifacts/stage1_ae_kolmogorov_crop_$TAG.pt \
  --prop-ckpt artifacts/stage2_prop_kolmogorov_crop_$TAG.pt --scale 1.0 --seed 0 --n-directions 384 \
  --tag ${TAG}_x1.0_seed0 > artifacts/logs/crop_eval_$TAG.log 2>&1
log "lyap $(grep -E 'rel_err_pct|model_D_KY' artifacts/logs/crop_eval_$TAG.log | tr -d '\n ')"
log "J18a DONE"
