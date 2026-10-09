#!/bin/bash
# Consolidation Phases 1+2 (docs/ks_local_field_ltransfer.tex sec:consolidation), 2026-10-05, user-approved:
# canonical recipe = C19 per-patch autoencoder (reach 0/0, c=6) + 2D patch training (40x40-site crops)
# + P9b long free rollouts (k_max 32, k_mid 16, 9000 steps, end noise 0.02).
#   TRAIN lane (one training at a time, MPS): p7b_ks2d, k_a005, p7c_ks2d (+spectral reg), k_a01, k_a0025
#   EVAL lane  (CPU): each model's evals as soon as it is trained; 2 IC seeds per size (user: trimmed from 3)
#   TRUTH lane (CPU): Kolmogorov alpha=0.05 1x truths, IC seeds 0-2, 120 time units, 40 directions
#                     (same ICs as the model evals, so native errors can be scored per IC)
# Normal priority. Replaces scripts/run_phase2_2dks_spectral_ablation.sh.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments; L=$A/logs; D=$A/datasets
log () { echo "[cons $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }

CANON="--crop-sites 40 --crop-sites-y 40 --n-offsets 16 --local-channels 6 --site-mix-radius 0 --n-site-mix-layers 0 \
--dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000"
# tag | dataset | extra flags
MODELS=(
  "p7b_ks2d|$D/ks2d_wide6.h5|"
  "k_a005|$D/kolmogorov_wide6x6_alpha0.05.h5|--y-align 32"
  "p7c_ks2d|$D/ks2d_wide6.h5|--w-spectral 1.0 --spectral-bands 0,1,1,999"
  "k_a01|$D/kolmogorov_wide6x6_alpha0.1.h5|--y-align 32"
  "k_a0025|$D/kolmogorov_wide6x6_alpha0.025.h5|--y-align 32"
)

train_lane () {
  for m in "${MODELS[@]}"; do IFS='|' read -r tag data extra <<< "$m"
    run $A/stage2_prop_kolmogorov_crop_$tag.pt "train $tag" sh -c "python scripts/train_kolmogorov_crop.py \
      --dataset $data --tag $tag $CANON $extra >> $L/crop_train_$tag.log 2>&1"
    touch $L/.train_attempted_$tag
  done; log "TRAIN LANE DONE"; }

ev_k () { tag=$1; sc=$2; s=$3; a=$4; ta=""
  case "$a" in
    0.05) [ $sc = 3 ] && ta="--truth-json $E/crop/truth_a005_x3_seed0.json" ;;
    0.025) ta="--truth-json $E/crop/truth_a0p025_x${sc}_seed0.json" ;;
  esac
  run $E/crop/${tag}_x${sc}_seed$s.json "$tag eval x$sc seed$s" sh -c "python scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s $ta >> $L/crop_eval_$tag.log 2>&1"; }
ev_ks () { tag=$1; sc=$2; s=$3; nd=""; [ "$sc" = "2.0" ] && nd="--n-directions 512"
  run $E/crop/${tag}_x${sc}_seed$s.json "$tag eval x$sc seed$s" sh -c "python scripts/exp_crop_eval_ks2d.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s $nd >> $L/crop_eval_$tag.log 2>&1"; }

eval_lane () {
  while pgrep -f run_phase0_c19_verify.sh > /dev/null; do sleep 60; done   # let Phase 0's control finish
  for m in "${MODELS[@]}"; do IFS='|' read -r tag data extra <<< "$m"
    while [ ! -f $L/.train_attempted_$tag ]; do sleep 60; done
    [ -f $A/stage2_prop_kolmogorov_crop_$tag.pt ] || { log "no checkpoint for $tag; skipping its evals"; continue; }
    for s in 0 1; do
      case $tag in
        *ks2d) for sc in 1.0 1.5 2.0; do ev_ks $tag $sc $s; done ;;
        k_a005) for sc in 1 2 3; do ev_k $tag $sc $s 0.05; done ;;
        k_a01) for sc in 1 2 3; do ev_k $tag $sc $s 0.1; done ;;
        k_a0025) for sc in 1 2 3; do ev_k $tag $sc $s 0.025; done ;;
      esac
    done
    if [ $tag = k_a005 ]; then
      run $E/rollout_stats/k_a005_seed0_n1000_y1.json "k_a005 rollout stats vs truth" sh -c \
        "python scripts/exp_rollout_stats_kolmogorov.py --tag k_a005 --scales 1,2,3 --n-steps 1000 --with-truth \
        >> $L/rollout_stats_k_a005.log 2>&1"
    fi
  done; log "EVAL LANE DONE"; }

truth_lane () {
  for s in 0 1 2; do
    run $E/crop/truth_a0p05_x1_seed${s}_T120.json "truth alpha=0.05 1x seed$s (T=120)" sh -c \
      "python scripts/run_truth_kolmogorov.py 0.05 1 $s 40 100 120 20 >> $L/truth_a005_x1_T120_seed$s.log 2>&1" &
  done; wait; log "TRUTH LANE DONE"; }

train_lane & eval_lane & truth_lane & wait
log "ALL DONE"
