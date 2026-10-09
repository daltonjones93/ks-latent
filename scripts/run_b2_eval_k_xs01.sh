#!/bin/bash
# B2 (docs/paper1_tasks.md): evaluate k_xs01 (Kolmogorov alpha=0.1, canonical recipe,
# trained B1) at 3 IC seeds x 3 sizes against the built-in alpha=0.1 truths
# (15.88/30.34/56.35, exp_crop_eval.py's own TRUTH table).
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments
log () { echo "[b2 $(date '+%H:%M:%S')] $*" >> $A/logs/b2_eval_k_xs01.log; }

run () {
  sc=$1; s=$2
  out=$E/crop/k_xs01_x${sc}_seed${s}.json
  if [ -f "$out" ]; then log "skip (done): x$sc seed$s"; return; fi
  log "running: x$sc seed$s"
  mamba run -n da_env python -u scripts/exp_crop_eval.py \
    --ae-ckpt $A/stage1_ae_kolmogorov_crop_k_xs01.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_k_xs01.pt \
    --scale $sc --seed $s --tag k_xs01_x${sc}_seed${s} \
    >> $A/logs/b2_eval_k_xs01.log 2>&1 \
  && log "done: x$sc seed$s" || log "FAILED: x$sc seed$s"
}

for sc in 1 2 3; do for s in 0 1 2; do run $sc $s; done; done
log "B2 ALL DONE"
