#!/bin/bash
# J17C (2026-10-10, user design): truth = standard 2D KS (the latent filter's existing 0.312 result, model
# p7c12d10_ks2d, needs no retraining); the physics filter's model = KS + a spurious -beta P[u'^3] term.
# Strength set by the model's one-cycle forecast error (scan: beta 1.1e-3 -> 0.078, 4.4e-3 -> 0.278,
# 1e-2 -> 0.538). Per level: tuning on seed 0, then remedies (nabla^2 estimation, cubic-coefficient
# estimation, bias correction +- Q); best on seed 0 -> seeds 1-2. Usage: run_j17c.sh <beta>
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
B=$1; LOG=artifacts/logs/j17c.log; E=artifacts/experiments/da2d; P=J17C_b${B}
log() { echo "$(date '+%F %T') $*" >> $LOG; }
ENV="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2"
pde() { tag=$1; shift
  [ -f $E/i11_${tag}.json ] && return
  env $ENV nice -n 10 $PY scripts/exp_i11_pde_letkf.py --workers 2 --model-cubic-coef $B --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  log "done $tag rc=$? $(grep -E "^$tag |Error" artifacts/logs/i11_${tag}.log | cut -c1-170)"; }
best() { $PY - "$1" <<'PYEOF'
import glob, json, math, sys
r = []
for f in glob.glob(f"artifacts/experiments/da2d/i11_{sys.argv[1]}.json"):
    e = json.load(open(f))["steady"].get("analysis")
    if e is not None and math.isfinite(e): r.append((e, f))
print(min(r)[1].split("i11_")[1][:-5] if r else "")
PYEOF
}
ME=$E/j16_model_error_mcub_$B.npz
[ -f $ME ] || env $ENV nice -n 10 $PY scripts/exp_j16_estimate_pde_model_error.py --model-cubic-coef $B --n-starts 100 --out mcub_$B >> $LOG 2>&1
lane() { for r in "$@"; do for c in 8.25 16; do pde ${P}_c${c}_rtps${r}_s0 --loc-c $c --rtps $r; done; done; }
lane 0.5 0.9 & lane 0.7 1.0 & wait
BT=$(best "${P}_c*_rtps*_s0"); C=$(echo $BT | sed -E 's/.*_c([0-9.]+)_rtps.*/\1/'); R=$(echo $BT | sed -E 's/.*_rtps([0-9.]+)_s0/\1/')
log "$P best tuning: $BT"
IS=$($PY -c "print(0.5 * $B)"); MS=$($PY -c "print(0.05 * $B)")
( pde ${P}_c${C}_rtps${R}_lap_s0 --loc-c $C --rtps $R --estimate-param lap --param-init-std 0.05
  pde ${P}_c${C}_rtps${R}_cub_s0 --loc-c $C --rtps $R --estimate-param cub --param-init-std $IS --param-min-std $MS ) &
( for q in 0 0.5 1; do pde ${P}_c${C}_rtps${R}_bc_q${q}_s0 --loc-c $C --rtps $R --model-error-file $ME --bias-correct --q-scale $q; done ) &
wait
BP=$(best "${P}_*_s0"); log "$P best physics config (seed 0): $BP"
ARGS=$($PY - "$BP" <<'PYEOF'
import json, sys
a = json.load(open(f"artifacts/experiments/da2d/i11_{sys.argv[1]}.json"))["args"]
o = f"--loc-c {a['loc_c']} --rtps {a['rtps']}"
if a.get("estimate_param", "none") != "none":
    o += f" --estimate-param {a['estimate_param']} --param-init-std {a['param_init_std']} --param-min-std {a['param_min_std']}"
if a.get("bias_correct"): o += f" --model-error-file {a['model_error_file']} --bias-correct --q-scale {a['q_scale']}"
print(o)
PYEOF
)
BASE=${BP%_s0}
pde ${BASE}_s1 $ARGS --seed 1 & pde ${BASE}_s2 $ARGS --seed 2 & wait
log "$P DONE"
