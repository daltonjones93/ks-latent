#!/bin/bash
# Consolidation Phase 0 (docs/ks_local_field_ltransfer.tex sec:consolidation, 2026-10-05): verify C19.
# Lane A: IC seeds 1,2 of c19_e0d0_c6 at 1/2/3x.  Lane B: training seed 1 (c19_e0d0_c6_s1) and the
# full-reach c=6 control (c19_e6d6_c6), each followed by seed-0 evals.  One training at a time. Normal priority.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments; T3=$E/crop/truth_a005_x3_seed0.json; D05=$A/datasets/kolmogorov_wide16_alpha0.05.h5
log () { echo "[p0 $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }
ev () { tag=$1; sc=$2; s=$3; ta=""; [ $sc = 3 ] && ta="--truth-json $T3"
  run $E/crop/${tag}_x${sc}_seed$s.json "$tag eval x$sc seed$s" sh -c "python scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt \
    --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s $ta >> $A/logs/crop_eval_$tag.log 2>&1"; }
laneA () { for s in 1 2; do for sc in 1 2 3; do ev c19_e0d0_c6 $sc $s; done; done; }
laneB () {
  run $A/stage2_prop_kolmogorov_crop_c19_e0d0_c6_s1.pt "train c19_e0d0_c6_s1" sh -c "python scripts/train_kolmogorov_crop.py \
    --dataset $D05 --tag c19_e0d0_c6_s1 --seed 1 --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 \
    --dec-n-site-mix-layers 0 --local-channels 6 >> $A/logs/crop_train_c19_e0d0_c6_s1.log 2>&1"
  run $A/stage2_prop_kolmogorov_crop_c19_e6d6_c6.pt "train c19_e6d6_c6 (control)" sh -c "python scripts/train_kolmogorov_crop.py \
    --dataset $D05 --tag c19_e6d6_c6 --local-channels 6 >> $A/logs/crop_train_c19_e6d6_c6.log 2>&1"
  for t in c19_e0d0_c6_s1 c19_e6d6_c6; do for sc in 1 2 3; do ev $t $sc 0; done; done; }
laneA & laneB & wait
log "PHASE 0 ALL DONE"
