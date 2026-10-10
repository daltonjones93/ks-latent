#!/bin/bash
# K10 (2026-10-10): small gap-filling runs for the Part II rewrite (Block K).
#  (b) one-cycle model error for the J1 cases (x-axis of the crossover plot) -- now, cheap
#  (d1) Jacobian sparsity for the Part II model -- now, cheap
#  (a) observation density, Part II model: strides 4/8/16, K=1/4, 44^2, seed 0; stride 16 also 300 cycles -- after J17H
#  (d2) Jacobian timing for the Part II model -- after I14 (needs a quiet machine)
#  (e) fields at 440^2 for the snapshot figure (K3f) -- after I14
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/k10.log; E=artifacts/experiments/da2d; TAG=p7c12d10_ks2d
log() { echo "$(date '+%F %T') $*" >> $LOG; }
log "K10 start"
for spec in "lap1.1:--model-lap-coef 1.1" "lap1.25:--model-lap-coef 1.25" "adv1:--model-advect-x 1" \
            "adv2:--model-advect-x 2" "M64:--model-nx 64"; do
  name=${spec%%:*}; args=${spec#*:}
  [ -f $E/j16_model_error_$name.npz ] || OMP_NUM_THREADS=1 nice -n 10 $PY scripts/exp_j16_estimate_pde_model_error.py $args --n-starts 100 --out $name >> $LOG 2>&1 &
done
KS_TAG=$TAG OMP_NUM_THREADS=2 $PY scripts/exp_f2_jacobian_sparsity.py >> $LOG 2>&1
wait
log "(b) one-cycle errors: $(for n in lap1.1 lap1.25 adv1 adv2 M64; do $PY -c "import json; d=json.load(open('$E/j16_model_error_$n.json')); print('$n', round(d['residual_rms'],3), end=' ')"; done)"
lat() { out=$1; shift; [ -f $E/h9_${out}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag $TAG --rtps 0.7 --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
  log "done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-200 | tr '\n' ' ')"; }
while pgrep -f "run_j17h_all.sh|run_j17h_rtps06.sh" > /dev/null; do sleep 300; done
log "(a) observation density start"
( lat K10a_st4_T4 --obs-stride 4 --n-tilings 4; lat K10a_st4_T1 --obs-stride 4 --n-tilings 1 ) &
( lat K10a_st16_T4 --obs-stride 16 --n-tilings 4; lat K10a_st16_T1 --obs-stride 16 --n-tilings 1 ) &
( lat K10a_st16_T4_300 --obs-stride 16 --n-tilings 4 --n-cycles 300 ) &
wait
log "(a) DONE (stride 8 = the existing D10/I33 runs)"
until grep -q "I14 DONE" artifacts/logs/i7_i16_i14_j5.log 2>/dev/null; do sleep 300; done
KS_TAG=$TAG OMP_NUM_THREADS=4 $PY scripts/exp_f4_jacobian_timing.py >> $LOG 2>&1
log "(d2) Jacobian timing done"
lat K10e_L440_T4_fields --obs-stride 8 --n-tilings 4 --L 440 --save-fields 3 --no-free
log "K10 DONE"
