#!/bin/bash
# J17G extended tuning (2026-10-11, user-approved). The J17G runs tuned the physics-based filter over
# c in {8.25, 16} x RTPS in {0.5, 0.7, 0.9, 1.0} only, and ran the true-model filter ("floor") at the
# wrong-model filter's setting, untuned (it then lost to the latent filter at delta = 0.2: 0.469 vs 0.341).
# This completes the grid that Paper II App. D states for every physics-based filter:
#   c in {8.25, 16, 24} x RTPS in {0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, and multiplicative inflation
#   1.1 / 1.2 without RTPS,
# for the wrong-model filter (seed 0; then the remedies -- a,b,g estimation, bias correction -- at the new
# best tuning if it changed; then the best configuration on seeds 1-2) and for the true-model filter
# (`true_` runs, same grid, no remedies needed; best on seeds 1-2).
# Usage: run_j17g_tune.sh <delta> [lanes]. Log: artifacts/logs/j17g_tune.log.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
EPS=$1; LANES=${2:-3}; LOG=artifacts/logs/j17g_tune.log; E=artifacts/experiments/da2d
NAME=gani$EPS; DS=artifacts/datasets/ks2d_${NAME}_wide6.h5; P=J17G_d$EPS; ME=$E/j16_model_error_$NAME.npz
log() { echo "$(date '+%F %T') [$P tune] $*" >> $LOG; }
STD=$($PY -c "import h5py; print(h5py.File('$DS')['metadata'].attrs['normalization_std'])")
ENV="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2"
pde() { tag=$1; shift
  [ -f $E/i11_${tag}.json ] && return
  env $ENV nice -n 10 $PY scripts/exp_i11_pde_letkf.py --workers 2 --truth-ganiso $EPS --norm-std $STD --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  log "done $tag rc=$? $(grep -E "^$tag |Error" artifacts/logs/i11_${tag}.log | cut -c1-160)"; }
best() { $PY - "$1" <<'PYEOF'
import glob, json, math, re, sys
r = []
for f in glob.glob("artifacts/experiments/da2d/i11_*.json"):
    tag = f.split("i11_")[1][:-5]
    if not re.fullmatch(sys.argv[1], tag):
        continue
    e = json.load(open(f))["steady"].get("analysis")
    if e is not None and math.isfinite(e): r.append((e, tag))
print(min(r)[1] if r else "")
PYEOF
}
args_of() { $PY - "$1" <<'PYEOF'
import json, sys
a = json.load(open(f"artifacts/experiments/da2d/i11_{sys.argv[1]}.json"))["args"]
o = f"--loc-c {a['loc_c']} --rtps {a['rtps']}"
if float(a.get("inflation", 1.0)) != 1.0: o += f" --inflation {a['inflation']}"
if a.get("model_ganiso") not in (None, 0, 0.0): o += f" --model-ganiso {a['model_ganiso']}"
if a.get("estimate_param", "none") != "none": o += f" --estimate-param {a['estimate_param']} --param-init-std {a['param_init_std']}"
if a.get("bias_correct"): o += f" --model-error-file {a['model_error_file']} --bias-correct --q-scale {a['q_scale']}"
print(o)
PYEOF
}
# job list for seed 0: wrong model (pde_) and true model (true_)
JOBS=()
for c in 8.25 16 24; do
  for r in 0 0.3 0.5 0.6 0.7 0.8 0.9 1.0; do
    JOBS+=("${P}_pde_c${c}_rtps${r}_s0|--loc-c $c --rtps $r")
    JOBS+=("${P}_true_c${c}_rtps${r}_s0|--loc-c $c --rtps $r --model-ganiso $EPS")
  done
  for i in 1.1 1.2; do
    JOBS+=("${P}_pde_c${c}_infl${i}_s0|--loc-c $c --rtps 0 --inflation $i")
    JOBS+=("${P}_true_c${c}_infl${i}_s0|--loc-c $c --rtps 0 --inflation $i --model-ganiso $EPS")
  done
done
log "start: ${#JOBS[@]} seed-0 settings (existing ones skipped), $LANES lanes"
run_lane() { k=$1; n=0; for j in "${JOBS[@]}"; do
  if [ $((n % LANES)) -eq $k ]; then tag=${j%%|*}; a=${j#*|}; pde $tag $a; fi; n=$((n+1)); done; }
for k in $(seq 0 $((LANES-1))); do run_lane $k & done; wait
# --- wrong model: remedies at the new best tuning, then best configuration on seeds 1-2 ---
BT=$(best "${P}_pde_c[0-9.]+_(rtps[0-9.]+|infl[0-9.]+)_s0"); log "best wrong-model tuning: $BT"
A=$(args_of $BT); B=${BT%_s0}
( pde ${B}_abg_s0 $A --estimate-param abg --param-init-std 0.05
  pde ${B}_abg_bc_q0_s0 $A --estimate-param abg --param-init-std 0.05 --model-error-file $ME --bias-correct --q-scale 0 ) &
( for q in 0 0.5 1; do pde ${B}_bc_q${q}_s0 $A --model-error-file $ME --bias-correct --q-scale $q; done ) &
wait
BP=$(best "${P}_pde_.*_s0"); log "best wrong-model configuration (seed 0): $BP"
AP=$(args_of $BP); BB=${BP%_s0}
pde ${BB}_s1 $AP --seed 1 & pde ${BB}_s2 $AP --seed 2 &
# --- true model: best tuning on seeds 1-2 ---
BTT=$(best "${P}_true_.*_s0"); log "best true-model tuning (seed 0): $BTT"
AT=$(args_of $BTT); BTB=${BTT%_s0}
pde ${BTB}_s1 $AT --seed 1 & pde ${BTB}_s2 $AT --seed 2 &
wait
$PY - "$BB" "$BTB" "$P" <<'PYEOF' >> $LOG
import json, sys
from pathlib import Path
E = Path("artifacts/experiments/da2d")
def v(t): return json.load(open(E / f"i11_{t}.json"))["steady"]["analysis"]
def lat(s):
    p = E / f"h9_{sys.argv[3]}_lat_T4_s{s}.json"
    return json.load(open(p))["steady"]["blend"] if p.exists() else float("nan")
w = [v(f"{sys.argv[1]}_s{s}") for s in range(3)]; t = [v(f"{sys.argv[2]}_s{s}") for s in range(3)]
l = [lat(s) for s in range(3)]
f = lambda x: " / ".join(f"{a:.3f}" for a in x) + f"  mean {sum(x)/3:.3f}"
print(f"  [{sys.argv[3]} tune] SUMMARY wrong model {sys.argv[1]}: {f(w)}")
print(f"  [{sys.argv[3]} tune] SUMMARY true model  {sys.argv[2]}: {f(t)}")
print(f"  [{sys.argv[3]} tune] SUMMARY latent T4: {f(l)}")
PYEOF
log "TUNE DONE"
