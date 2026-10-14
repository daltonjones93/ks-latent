#!/bin/bash
# A2 (2026-10-14): the reach-3 window model retrained with longer Stage B rollouts (k-max 32, k-mid 16, 9000 steps;
# tag a2c_r3_k32) runs in free rollout (field std 1.45 vs 1.46 for c19_e0d0_c6), where the default k-max 12 recipe
# (a2_c19_e0d0_c6_r3, a2b_r3_seed1) settles into a large-amplitude state. Same evaluation as run_a2_reach3.sh:
# 1/2/3x on ICs 0-2 like c19_e0d0_c6. Low priority, two evals at a time.
cd "$(dirname "$0")/.."
A=artifacts; E=$A/experiments; PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
TAG=a2c_r3_k32
log () { echo "$(date '+%F %T') [A2C] $*" >> $A/logs/r4.log; }
ev () { sc=$1; s=$2; ta=""; [ $sc = 3 ] && [ -f $E/crop/truth_a005_x3_seed$s.json ] && ta="--truth-json $E/crop/truth_a005_x3_seed$s.json"
  [ -f $E/crop/${TAG}_x${sc}_seed$s.json ] && return
  taskpolicy -b $PY scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$TAG.pt --prop-ckpt $A/stage2_prop_kolmogorov_crop_$TAG.pt \
    --scale $sc --seed $s --tag ${TAG}_x${sc}_seed$s $ta >> $A/logs/crop_eval_$TAG.log 2>&1
  log "eval $TAG x$sc seed$s rc=$? $($PY -c "import json;d=json.load(open('$E/crop/${TAG}_x${sc}_seed$s.json'));print('D_KY',round(d['model_D_KY'],1),'n+',d['model_n_positive'],'lam1',round(d['model_lambda1'],3),'err%',d.get('rel_err_pct'))" 2>&1 | tail -1)"; }
log "A2C evals start"
for s in 0 1 2; do ev 1 $s; done
for s in 0 1 2; do ev 2 $s & ev 3 $s & wait; done
log "A2C DONE"
