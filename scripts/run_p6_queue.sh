#!/bin/bash
# C16 + C13 diagnostics, then P6 (2026-10-03): same-domain patch training at alpha=0.05. Train on windows of a
# 3x2pi periodic domain, deploy on the full 3x domain, compare with the 3x truth.
# Narrow encoder (radius 1 x 2 layers) + window-1 propagator so crops can be much
# smaller than the 24-site domain. Free-interior sweep {6, 8, 12} sites
# (0.75, 1, 1.5 box widths) plus the full-periodic upper bound. Waits for the C8/C9
# queue; one task at a time; every task skips if its output exists.
cd "$(dirname "$0")/.."
while pgrep -f run_c8_c9_queue.sh > /dev/null; do sleep 60; done
A=artifacts; E=$A/experiments
DATA=$A/datasets/kolmogorov_wide3_alpha0.05.h5
TRUTH=$E/crop/truth_a005_x3_seed0.json
COMMON="--window 1 --site-mix-radius 1 --n-site-mix-layers 2 --n-offsets 16"
log () { echo "[p6 $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }

# Staged first (2026-10-03, user: "queue C16 before P6"; highest-likelihood-first): the two
# cheap no-training diagnostics C16 (self-similarity) and C13 (large-scale closure).
run $E/c16/c16_result.json "C16 self-similarity diagnostic" python scripts/exp_c16_self_similarity.py \
  >> $A/logs/c16.log 2>&1
run $E/c13/c13_result.json "C13 closure diagnostic" python scripts/exp_c13_closure_diagnostic.py \
  >> $A/logs/c13.log 2>&1

run $DATA "P6 data (3x, alpha=0.05)" python scripts/gen_kolmogorov_wide_dataset.py --alpha 0.05 --x-scale 3 \
  --n-train 12 --n-val 2 --seed-base 700 --out $DATA >> $A/logs/p6_data.log 2>&1

for arm in "p6_i6:14:" "p6_i8:16:" "p6_i12:20:" "p6_full:24:--periodic-full"; do
  tag=${arm%%:*}; rest=${arm#*:}; crop=${rest%%:*}; extra=${rest#*:}
  run $A/stage2_prop_kolmogorov_crop_$tag.pt "$tag train (crop $crop)" \
    python scripts/train_kolmogorov_crop.py --dataset $DATA --tag $tag --crop-sites $crop $COMMON $extra \
      >> $A/logs/crop_train_$tag.log 2>&1
  for s in 0 1 2; do
    run $E/crop/${tag}_x3_seed$s.json "$tag eval x3 seed$s" \
      python scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt \
        --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt --scale 3 --seed $s --tag ${tag}_x3_seed$s \
        --truth-json $TRUTH >> $A/logs/crop_eval_p6.log 2>&1
  done
done
log "ALL DONE"
