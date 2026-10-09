#!/bin/bash
# v3 queue (2026-10-04, user: prioritize fixing P9). P9 drifts in free rollout even in its own (tall)
# geometry -> training-stability problem. P9b: same data, same autoencoder (copied from P9), propagator
# retrained with longer free rollouts (k_max 12 -> 32, k_mid 8 -> 16), more steps, stronger end noise.
# Normal priority for P9b (user priority); then C19 c6 evals and P8 at background QoS.
# NOTE: per-job redirects wrap the inner command in sh -c so run()'s own log lines reach THIS log.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments; T3=$E/crop/truth_a005_x3_seed0.json
log () { echo "[q3 $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }
evalk () { tag=$1; pri=$2; seeds=$3; for sc in 1 2 3; do for s in $seeds; do ta=""; [ $sc = 3 ] && ta="--truth-json $T3"
  while pgrep -f "exp_crop_eval.py.*--tag ${tag}_x${sc}_seed$s" > /dev/null; do sleep 30; done
  run $E/crop/${tag}_x${sc}_seed$s.json "$tag eval x$sc seed$s" sh -c "$pri python scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s $ta >> $A/logs/crop_eval_$tag.log 2>&1"; done; done; }

# P9b (one training at a time: nothing else is training now)
run $A/stage2_prop_kolmogorov_crop_p9b_a005.pt "P9b train (k_max 32)" sh -c "python scripts/train_kolmogorov_crop.py \
  --dataset $A/datasets/kolmogorov_wide6x6_alpha0.05.h5 --tag p9b_a005 --crop-sites 40 --crop-sites-y 40 --y-align 32 \
  --n-offsets 16 --k-max 32 --k-mid 16 --noise-end 0.05 --prop-steps 9000 >> $A/logs/crop_train_p9b_a005.log 2>&1"
if [ -f $A/stage2_prop_kolmogorov_crop_p9b_a005.pt ]; then
  run $E/rollout_stats/p9b_a005_seed0_n1000_y1.json "P9b rollout stats 1/2/3x" sh -c "python scripts/exp_rollout_stats_kolmogorov.py \
    --tag p9b_a005 --scales 1,2,3 --n-steps 1000 >> $A/logs/rollout_stats_p9b.log 2>&1"
  evalk p9b_a005 "" "0 1 2"
fi

# C19 c6 evals (background), then P8
evalk c19_e0d0_c6 "taskpolicy -b" "0"
D025=$A/datasets/kolmogorov_wide16_alpha0.025.h5
run $A/stage2_prop_kolmogorov_crop_p8_a0025.pt "P8 (alpha=0.025) train" sh -c "taskpolicy -b python scripts/train_kolmogorov_crop.py \
  --dataset $D025 --tag p8_a0025 >> $A/logs/crop_train_p8_a0025.log 2>&1"
for sc in 1 2 3; do for s in 0 1 2; do
  run $E/crop/p8_a0025_x${sc}_seed$s.json "P8 eval x$sc seed$s" sh -c "taskpolicy -b python scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_p8_a0025.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_p8_a0025.pt \
    --scale $sc --seed $s --tag p8_a0025_x${sc}_seed$s --truth-json $E/crop/truth_a0p025_x${sc}_seed0.json \
    >> $A/logs/crop_eval_p8.log 2>&1"
done; done
log "ALL DONE"
