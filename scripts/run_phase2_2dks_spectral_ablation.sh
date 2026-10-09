#!/bin/bash
# Phase 2 (docs/ks_local_field_ltransfer.tex sec:consolidation), 2D KS spectral-regularizer
# ablation (user, 2026-10-05): retrain P7 with the canonical fixes (C19 per-patch c=6
# autoencoder + P9b-style long free rollouts, noise held at baseline per sec:consolidation),
# with and without the spectral-band-share regularizer (sec:propagator_lessons item 3).
# Waits for Phase 0 (C19 verify) and the Phase 1 datasets to clear first (shared CPU).
# One job at a time, normal priority.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments; D=$A/datasets/ks2d_wide6.h5
log () { echo "[p2ks $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }
ev () { tag=$1; sc=$2; s=$3; nd=""; [ "$sc" = "2.0" ] && nd="--n-directions 512"
  run $E/crop/${tag}_x${sc}_seed$s.json "$tag eval x$sc seed$s" \
    python scripts/exp_crop_eval_ks2d.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt \
      --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s $nd \
      >> $A/logs/crop_eval_$tag.log 2>&1; }

while pgrep -f "run_phase0_c19_verify.sh" > /dev/null || pgrep -f "gen_kolmogorov_wide_dataset.py --alpha 0.1" > /dev/null \
      || pgrep -f "gen_kolmogorov_wide_dataset.py --alpha 0.025" > /dev/null; do sleep 60; done
log "Phase 0 / Phase 1 data clear; starting"

COMMON="--crop-sites 40 --crop-sites-y 40 --n-offsets 16 --local-channels 6 --site-mix-radius 0 \
--n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --k-max 32 --k-mid 16 --noise-end 0.02"

run $A/stage2_prop_kolmogorov_crop_p7b_ks2d.pt "p7b (canonical, no spectral reg) train" \
  python scripts/train_kolmogorov_crop.py --dataset $D --tag p7b_ks2d $COMMON \
    >> $A/logs/crop_train_p7b_ks2d.log 2>&1
for sc in 1.0 1.5 2.0; do for s in 0 1 2; do ev p7b_ks2d $sc $s; done; done

run $A/stage2_prop_kolmogorov_crop_p7c_ks2d.pt "p7c (canonical + spectral reg) train" \
  python scripts/train_kolmogorov_crop.py --dataset $D --tag p7c_ks2d $COMMON \
    --w-spectral 1.0 --spectral-bands "0,1,1,999" >> $A/logs/crop_train_p7c_ks2d.log 2>&1
for sc in 1.0 1.5 2.0; do for s in 0 1 2; do ev p7c_ks2d $sc $s; done; done

log "ALL DONE"
