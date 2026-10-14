#!/bin/bash
# A2 diagnosis (2026-10-14): the reach-3 window model a2_c19_e0d0_c6_r3 free-runs to a large-amplitude
# state (field std 3.4 vs 1.46 for c19_e0d0_c6, D_KY 0, lambda1 -0.08) although its training loss is fine.
# Variants, each from the copied c19_e0d0_c6 Stage A (so only Stage B changes), rollout statistics at 1x:
#   a2b: same recipe, seed 1 (is it a bad draw?)
#   a2c: longer rollouts in Stage B (k-max 32, k-mid 16, 9000 steps, as the R4 recipes)
cd "$(dirname "$0")/.."
A=artifacts; PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
D=$A/datasets/kolmogorov_wide16_alpha0.05.h5
log () { echo "$(date '+%F %T') [A2DIAG] $*" >> $A/logs/r4.log; }
run () { tag=$1; shift
  [ -f $A/stage1_ae_kolmogorov_crop_$tag.pt ] || cp $A/stage1_ae_kolmogorov_crop_c19_e0d0_c6.pt $A/stage1_ae_kolmogorov_crop_$tag.pt
  log "train $tag start ($*)"
  taskpolicy -b $PY scripts/train_kolmogorov_crop.py --dataset $D --tag $tag \
    --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --local-channels 6 \
    --prop-layers 1 --window 3 "$@" > $A/logs/crop_train_$tag.log 2>&1
  log "train $tag rc=$? $(tail -2 $A/logs/crop_train_$tag.log | head -1 | cut -c1-120)"
  taskpolicy -b $PY scripts/exp_rollout_stats_kolmogorov.py --tag $tag --scales 1 --n-steps 1000 \
    > $A/logs/a2diag_rollout_$tag.log 2>&1
  log "rollout $tag $(tail -1 $A/logs/a2diag_rollout_$tag.log | cut -c1-330)"; }
run a2b_r3_seed1 --seed 1
run a2c_r3_k32 --k-max 32 --k-mid 16 --prop-steps 9000
log "A2DIAG DONE"
