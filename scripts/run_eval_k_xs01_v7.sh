#!/bin/bash
# Evaluate k_xs01_v7 (k_max=16, w_logdet=0.006, w_var=0.005, w_rank=0.0, logdet_eps=0)
# at 3 IC seeds x 3 sizes. Isolates logdet_eps from k_max (v5/v6a used default eps=1e-3).
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments
log () { echo "[evalv7 $(date '+%H:%M:%S')] $*" >> $A/logs/eval_k_xs01_v7.log; }

run () {
  sc=$1; s=$2
  out=$E/crop/k_xs01_v7_x${sc}_seed${s}.json
  if [ -f "$out" ]; then log "skip (done): x$sc seed$s"; return; fi
  log "running: x$sc seed$s"
  mamba run -n da_env python -u scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_k_xs01_v7.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_k_xs01_v7.pt \
    --scale $sc --seed $s --tag k_xs01_v7_x${sc}_seed${s} \
    >> $A/logs/eval_k_xs01_v7.log 2>&1 \
  && log "done: x$sc seed$s" || log "FAILED: x$sc seed$s"
}

for sc in 1 2 3; do for s in 0 1 2; do run $sc $s; done; done
log "EVAL k_xs01_v7 ALL DONE"
