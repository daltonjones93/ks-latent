#!/bin/bash
# Memory re-measurement after the in-analysis autograd fix (2026-10-08).
# Skill/RMSE are unchanged by the fix (verified to 3e-8); only peak memory
# changes. Re-runs the N=64 configurations used in the memory panel
# (Fig. 9f) under new "_mem" tags. Starts after run_da2d_rerun.sh finishes.
cd "$(dirname "$0")/.."
OUT=artifacts/experiments/da2d; LOG=artifacts/logs/da2d_rerun.log
AE=artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt; PR=artifacts/stage2_prop_kolmogorov_crop_p7b_ks2d.pt
log () { echo "[da2d-mem $(date '+%H:%M:%S')] $*" >> $LOG; }
until grep -q "ALL DONE" $LOG; do sleep 60; done

run () {
  tag=$1; shift
  if [ -f "$OUT/da2d_${tag}.json" ]; then log "skip (done): $tag"; return; fi
  log "running: $tag $*"
  mamba run -n da_env python scripts/run_da_pff_2d_ltransfer.py \
    --ae-checkpoint $AE --prop-checkpoint $PR --n-cycles 100 --obs-stride 8 "$@" --tag "$tag" \
    > "$OUT/da2d_${tag}.stdout" 2>&1 && log "done: $tag" || log "FAILED: $tag"
}

for L in 44 66 88 220; do
  run "M_L${L}_local_N64_s0" --L $L --n-ensemble 64 --method letkf_local2d --seed 0
  run "M_L${L}_denseGC_N64_s0" --L $L --n-ensemble 64 --method letkf --seed 0
done
log "MEMFIX DONE"
