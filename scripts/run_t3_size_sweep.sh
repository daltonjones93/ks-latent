#!/bin/bash
# T3 experiment (docs/ks_local_field_ltransfer.tex sec:c19_da_hypotheses,
# user-directed 2026-10-06: "launch T3"). Same comparison as T2
# (dense `letkf` vs per-site `letkf_local2d`, same checkpoint, same seed,
# same N_ens/cycles/spinup) repeated at the two transfer sizes T2 didn't
# cover -- this is the one that actually tests H2 (does dense cost/memory
# blow up faster than local as domain size grows).
cd "$(dirname "$0")/.."
A=artifacts
log () { echo "[t3 $(date '+%H:%M:%S')] $*" >> $A/logs/t3_sweep.log; }  # fixed 2026-10-06 (task F1): used to
# go to stdout, silently swallowed by the top-level `nohup ... > /dev/null` wrapper these
# scripts are launched under -- the orchestrator's own "running:"/"done:"/"FAILED:" lines
# never reached the log file at all, only the python subprocess's own output (redirected
# separately) did. Same fix already applied in run_t4_inflation_sweep.sh/run_t5_long_cycles.sh.
run () {
  L=$1; method=$2; tag=t3_${method}_L${L%.*}
  out=$A/da_pff_2d_ltransfer_${tag}.json
  if [ -f "$out" ]; then log "skip (done): $tag"; return; fi
  log "running: $method at L=$L"
  /usr/bin/time -p mamba run -n da_env python -u scripts/run_da_pff_2d_ltransfer.py \
    --ae-checkpoint $A/stage1_ae_kolmogorov_crop_p7b_ks2d.pt \
    --prop-checkpoint $A/stage2_prop_kolmogorov_crop_p7b_ks2d.pt \
    --method $method --old-L 44.0 --new-L $L --spinup-time 200.0 \
    --n-ensemble 64 --n-cycles 20 --n-prop-steps 5 --seed 42 --tag $tag \
    >> $A/logs/t3_sweep.log 2>&1 \
  && log "done: $tag" || log "FAILED: $tag"
}

for L in 66.0 88.0; do
  run $L letkf
  run $L letkf_local2d
done
log "ALL DONE"
