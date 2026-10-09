#!/bin/bash
# Wave 1 (docs/paper1_tasks.md, user-directed 2026-10-06: "do as much as I can as
# quickly as possible"): the fast (~10-20 min each) background DA experiments that
# build directly on T1-T5 without new code. All at native size (44^2), 100 cycles
# (per T5: 20 cycles understates steady-state skill for every run), same checkpoint
# and seed as every prior run this session.
cd "$(dirname "$0")/.."
A=artifacts
log () { echo "[w1 $(date '+%H:%M:%S')] $*" >> $A/logs/wave1_fast_da.log; }

run () {
  tag=$1; shift
  out=$A/da_pff_2d_ltransfer_${tag}.json
  if [ -f "$out" ]; then log "skip (done): $tag"; return; fi
  log "running: $tag"
  /usr/bin/time -p mamba run -n da_env python -u scripts/run_da_pff_2d_ltransfer.py \
    --ae-checkpoint $A/stage1_ae_kolmogorov_crop_p7b_ks2d.pt \
    --prop-checkpoint $A/stage2_prop_kolmogorov_crop_p7b_ks2d.pt \
    --old-L 44.0 --new-L 44.0 --spinup-time 200.0 --n-cycles 100 --n-prop-steps 5 --seed 42 \
    --tag $tag "$@" \
    >> $A/logs/wave1_fast_da.log 2>&1 \
  && log "done: $tag" || log "FAILED: $tag"
}

# E6: unlocalized baseline, matched N_ens=64
run w1_e6_unlocalized --method letkf --n-ensemble 64 --localizer none

# E1a-c: model-noise sweep (0.08 = T5's existing baseline, already have it)
run w1_e1_noise0.04 --method letkf --n-ensemble 64 --model-noise-std 0.04
run w1_e1_noise0.02 --method letkf --n-ensemble 64 --model-noise-std 0.02
run w1_e1_noise0.01 --method letkf --n-ensemble 64 --model-noise-std 0.01

# E4a-d: ensemble-size sweep, local filter
run w1_e4_nens16  --method letkf_local2d --n-ensemble 16
run w1_e4_nens32  --method letkf_local2d --n-ensemble 32
run w1_e4_nens64  --method letkf_local2d --n-ensemble 64
run w1_e4_nens128 --method letkf_local2d --n-ensemble 128

log "WAVE 1 ALL DONE"
