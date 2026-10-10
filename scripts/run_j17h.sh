#!/bin/bash
# J17H (2026-10-10, user design): truth = standard 2D KS; the physics filter's model is structurally wrong
# (damping order p != 4) but has ALL coefficients estimable:  u_t = -a lap u - b (-lap)^{p/2} u - (g/2)|grad u|^2.
# Starts from the offline best fit (exp_j17h_fit.py, seed 3). Per level: tuning (seed 0, fixed best-fit params),
# then remedies at the best tuning: joint online (a,b,g) estimation; bias correction +- Q; both. Best on seed 0
# -> seeds 1-2. Latent comparison: existing p7c12d10_ks2d K=4 result (0.312, seeds 0-2). Usage: run_j17h.sh <p>
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
O=$1; LOG=artifacts/logs/j17h.log; E=artifacts/experiments/da2d; P=J17H_p${O}
log() { echo "$(date '+%F %T') $*" >> $LOG; }
read A B G <<< "$($PY -c "import json; d=json.load(open('$E/j17h_fit_order$O.json')); print(d['a'], d['b'], d['g'])")"
MODEL="--model-order $O --model-lap-coef $A --model-hyper-coef $B --model-nl-coef $G"
log "$P start: best fit a=$A b=$B g=$G"
ENV="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2"
pde() { tag=$1; shift
  [ -f $E/i11_${tag}.json ] && return
  env $ENV nice -n 10 $PY scripts/exp_i11_pde_letkf.py --workers 2 $MODEL --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  log "done $tag rc=$? $(grep -E "^$tag |Error" artifacts/logs/i11_${tag}.log | cut -c1-230)"; }
best() { $PY - "$1" <<'PYEOF'
import glob, json, math, sys
r = []
for f in glob.glob(f"artifacts/experiments/da2d/i11_{sys.argv[1]}.json"):
    e = json.load(open(f))["steady"].get("analysis")
    if e is not None and math.isfinite(e): r.append((e, f))
print(min(r)[1].split("i11_")[1][:-5] if r else "")
PYEOF
}
ME=$E/j16_model_error_hyp$O.npz
[ -f $ME ] || env $ENV nice -n 10 $PY scripts/exp_j16_estimate_pde_model_error.py $MODEL --n-starts 100 --out hyp$O >> $LOG 2>&1
lane() { for r in "$@"; do for c in 8.25 16; do pde ${P}_c${c}_rtps${r}_s0 --loc-c $c --rtps $r; done; done; }
lane 0.5 0.9 & lane 0.7 1.0 & wait
BT=$(best "${P}_c*_rtps*_s0"); C=$(echo $BT | sed -E 's/.*_c([0-9.]+)_rtps.*/\1/'); R=$(echo $BT | sed -E 's/.*_rtps([0-9.]+)_s0/\1/')
log "$P best tuning: $BT"
EST="--estimate-param abg --param-init-std 0.05 --param-min-std 0.005"
( pde ${P}_c${C}_rtps${R}_abg_s0 --loc-c $C --rtps $R $EST
  pde ${P}_c${C}_rtps${R}_abg_bc_q0_s0 --loc-c $C --rtps $R $EST --model-error-file $ME --bias-correct --q-scale 0 ) &
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
