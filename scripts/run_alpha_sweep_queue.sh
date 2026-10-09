#!/bin/bash
# Drag sweep (2026-10-03): alpha = 0.025 replicate of the P2 protocol, plus the alpha = 0
# saturation pilot that gates any drag-free runs. Waits for the C16/C13/P6 queue.
# The three alpha=0.025 truth runs (single-threaded, long) start immediately in the
# background; everything else runs one task at a time. Rerunnable: tasks skip if done.
cd "$(dirname "$0")/.."
sleep 5  # let P7's queue start first
while pgrep -f run_p6_queue.sh > /dev/null || pgrep -f run_p7_step2_queue.sh > /dev/null; do sleep 60; done
A=artifacts; E=$A/experiments
DATA=$A/datasets/kolmogorov_wide16_alpha0.025.h5
log () { echo "[sweep $(date '+%m-%d %H:%M:%S')] $*"; }
run () { out=$1; shift; name=$1; shift
  if [ -f "$out" ]; then log "skip (done): $name"; return; fi
  log "running: $name"; "$@" && log "done: $name" || log "FAILED: $name"; }

for spec in "1 64" "2 128" "3 176"; do set -- $spec
  [ -f $E/crop/truth_a0p025_x$1_seed0.json ] || \
    (python scripts/run_truth_kolmogorov.py 0.025 $1 0 $2 >> $A/logs/truth_a0p025_x$1.log 2>&1 &)
done
log "launched alpha=0.025 truth runs (1x/2x/3x) in background"

run $E/alpha0/alpha0_pilot.json "alpha=0 saturation pilot" python scripts/exp_alpha0_pilot.py >> $A/logs/alpha0_pilot.log 2>&1
run $DATA "alpha=0.025 wide data" python scripts/gen_kolmogorov_wide_dataset.py --alpha 0.025 --out $DATA \
  >> $A/logs/p0_wide_a0.025.log 2>&1
run $A/stage2_prop_kolmogorov_crop_p8_a0025.pt "P8 (alpha=0.025) train" \
  python scripts/train_kolmogorov_crop.py --dataset $DATA --tag p8_a0025 >> $A/logs/crop_train_p8_a0025.log 2>&1
for sc in 1 2 3; do for s in 0 1 2; do
  run $E/crop/p8_a0025_x${sc}_seed$s.json "P8 eval x$sc seed$s" \
    python scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_p8_a0025.pt \
      --prop-ckpt $A/stage2_prop_kolmogorov_crop_p8_a0025.pt --scale $sc --seed $s --tag p8_a0025_x${sc}_seed$s \
      --truth-json $E/crop/truth_a0p025_x${sc}_seed0.json >> $A/logs/crop_eval_p8.log 2>&1
done; done
log "ALL DONE (truth runs may still be running in background)"
