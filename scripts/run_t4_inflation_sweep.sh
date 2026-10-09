#!/bin/bash
# T4: LETKF inflation sweep (user-directed 2026-10-06, after noticing DA
# skill was modest ~1.7-2.0x at every size/method tried in T2/T3: "not
# sure how to interpret it honestly"). Every T2/T3 run used the literal
# default letkf_inflation=1.0 -- no covariance inflation at all, a
# well-known common cause of weak EnKF/LETKF skill (ensemble spread
# collapses over cycles, filter becomes overconfident). Isolates this
# one variable at native size (cheap, ~8-9 min/run): if skill improves
# meaningfully with inflation, weak skill was a tuning gap, not a
# limitation of the architecture/approach. Waits for the currently
# running dense-LETKF-at-88^2 retry (pid passed as $1) to finish first,
# so only one job uses the machine at a time.
cd "$(dirname "$0")/.."
A=artifacts
log () { echo "[t4 $(date '+%H:%M:%S')] $*" >> $A/logs/t4_inflation_sweep.log; }

WAIT_PID=$1
if [ -n "$WAIT_PID" ]; then
  log "waiting for pid $WAIT_PID (dense@88^2 retry) to finish"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
  log "pid $WAIT_PID done, starting inflation sweep"
fi

run () {
  infl=$1; tag=t4_infl_${infl}
  out=$A/da_pff_2d_ltransfer_${tag}.json
  if [ -f "$out" ]; then log "skip (done): $tag"; return; fi
  log "running: inflation=$infl (native size, dense letkf)"
  /usr/bin/time -p mamba run -n da_env python -u scripts/run_da_pff_2d_ltransfer.py \
    --ae-checkpoint $A/stage1_ae_kolmogorov_crop_p7b_ks2d.pt \
    --prop-checkpoint $A/stage2_prop_kolmogorov_crop_p7b_ks2d.pt \
    --method letkf --old-L 44.0 --new-L 44.0 --spinup-time 200.0 \
    --n-ensemble 64 --n-cycles 20 --n-prop-steps 5 --seed 42 \
    --letkf-inflation $infl --tag $tag \
    >> $A/logs/t4_inflation_sweep.log 2>&1 \
  && log "done: $tag" || log "FAILED: $tag"
}

for infl in 1.0 1.05 1.1 1.2 1.5; do
  run $infl
done
log "ALL DONE"
