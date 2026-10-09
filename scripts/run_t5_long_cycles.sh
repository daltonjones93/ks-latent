#!/bin/bash
# T5: does DA skill plateau, or just needs a longer run? (user-directed
# 2026-10-06, following T4's inflation=1.0 per-cycle breakdown: skill
# climbed from ~1x to 3x over just 20 cycles with no sign of leveling
# off, and calibration was still settling toward 1.0 at the last cycle
# -- 20 cycles isn't enough to tell whether skill plateaus below the
# brief's own 6x target or keeps climbing past it.) Same checkpoint,
# seed, inflation=1.0 (the plain baseline, isolating "does more run
# length help" from the inflation question T4 is also testing), native
# size, 5x the cycle count (100 vs 20) so the per-cycle trace can
# actually show where (or whether) it levels off.
cd "$(dirname "$0")/.."
A=artifacts
log () { echo "[t5 $(date '+%H:%M:%S')] $*" >> $A/logs/t5_long_cycles.log; }

WAIT_PID=$1
if [ -n "$WAIT_PID" ]; then
  log "waiting for pid $WAIT_PID (t4 inflation sweep) to finish"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
  log "pid $WAIT_PID done, starting long-cycle run"
fi

tag=t5_long100_infl1.0
out=$A/da_pff_2d_ltransfer_${tag}.json
if [ -f "$out" ]; then
  log "skip (done): $tag"
else
  log "running: n_cycles=100, inflation=1.0, native size, dense letkf"
  /usr/bin/time -p mamba run -n da_env python -u scripts/run_da_pff_2d_ltransfer.py \
    --ae-checkpoint $A/stage1_ae_kolmogorov_crop_p7b_ks2d.pt \
    --prop-checkpoint $A/stage2_prop_kolmogorov_crop_p7b_ks2d.pt \
    --method letkf --old-L 44.0 --new-L 44.0 --spinup-time 200.0 \
    --n-ensemble 64 --n-cycles 100 --n-prop-steps 5 --seed 42 \
    --letkf-inflation 1.0 --tag $tag \
    >> $A/logs/t5_long_cycles.log 2>&1 \
  && log "done: $tag" || log "FAILED: $tag"
fi
log "ALL DONE"
