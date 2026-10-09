#!/bin/bash
# User-directed (2026-10-06): the clean A/B (k_xs005 vs k_xs005_noreg) showed the spectral
# regularizer does NOT fix the pervasive D_KY underestimation (no-reg was BETTER at 3x:
# 5.1% vs 9.9% mean error) -- see sec:k_xs005_ab_results. User hypothesis instead: training
# diversity is the real lever, specifically n_train (only 4 independent trajectories; the
# 32 x-offsets per trajectory give local, not independent, diversity) and --n-offsets.
# Datasets expanded 4->8 train trajectories (+2 val) via gen_kolmogorov_wide_dataset.py
# --n-train 8 --n-val 2 (reuses existing traj0-4, generates traj5-9). This trains NEW
# no-regularizer x-strip+long-rollout models on the expanded data with --n-offsets 64
# (doubled from 32), for both alpha=0.05 and alpha=0.025 -- testing the diversity
# hypothesis cleanly, without the already-tested-and-found-wanting regularizer confound.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments
T3=$E/crop/truth_a005_x3_seed0.json
log () { echo "[kxsexp $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }
ev005 () { tag=$1; sc=$2; s=$3; ta=""; [ $sc = 3 ] && ta="--truth-json $T3"
  run $E/crop/${tag}_x${sc}_seed$s.json "$tag eval x$sc seed$s" sh -c "python scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s $ta >> $A/logs/crop_eval_$tag.log 2>&1"; }
ev0025 () { tag=$1; sc=$2; s=$3
  run $E/crop/${tag}_x${sc}_seed$s.json "$tag eval x$sc seed$s" sh -c "python scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s --truth-json $E/crop/truth_a0p025_x${sc}_seed0.json \
    >> $A/logs/crop_eval_$tag.log 2>&1"; }

while pgrep -f "gen_kolmogorov_wide_dataset.py --alpha 0.05 --n-train 8" > /dev/null \
      || pgrep -f "gen_kolmogorov_wide_dataset.py --alpha 0.025 --n-train 8" > /dev/null; do sleep 60; done
log "expanded datasets ready"

BASE="--site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 \
--local-channels 6 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000 --n-offsets 64"

run $A/stage2_prop_kolmogorov_crop_k_xs005_v2.pt "train k_xs005_v2 (expanded data, no reg)" sh -c \
  "python scripts/train_kolmogorov_crop.py --dataset $A/datasets/kolmogorov_wide16_alpha0.05.h5 \
   --tag k_xs005_v2 $BASE >> $A/logs/crop_train_k_xs005_v2.log 2>&1"
for sc in 1 2 3; do for s in 0 1 2; do ev005 k_xs005_v2 $sc $s; done; done

run $A/stage2_prop_kolmogorov_crop_k_xs0025_v2.pt "train k_xs0025_v2 (expanded data, no reg)" sh -c \
  "python scripts/train_kolmogorov_crop.py --dataset $A/datasets/kolmogorov_wide16_alpha0.025.h5 \
   --tag k_xs0025_v2 $BASE >> $A/logs/crop_train_k_xs0025_v2.log 2>&1"
for sc in 1 2 3; do for s in 0 1 2; do ev0025 k_xs0025_v2 $sc $s; done; done

log "ALL DONE"
