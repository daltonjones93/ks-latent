#!/bin/bash
# J2 (fair tuning of the PDE filter, seed 0) then J1 (seeds 1-2 at each case's best setting).
# Low priority (nice 10, 2 threads) so the J4/I33 evaluation always runs first. Two lanes.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/j12.log
pde() { tag=$1; shift
  [ -f artifacts/experiments/da2d/i11_${tag}.json ] && return
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 nice -n 19 \
    $PY scripts/exp_i11_pde_letkf.py --workers 2 --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/i11_${tag}.log | cut -c1-170)" >> $LOG; }
# case name : base tag : model-error args
CASES=("lap1.1:I28W_lap1.1:--model-lap-coef 1.1" "lap1.25:I28W_lap1.25:--model-lap-coef 1.25"
       "adv1:I28W_adv1:--model-advect-x 1" "adv2:I28W_adv2:--model-advect-x 2"
       "frc0.2:I28W_frc0.2:--model-forcing-amp 0.2" "frc0.4:I28W_frc0.4:--model-forcing-amp 0.4"
       "M64:I28R_M64_pde:--model-nx 64" "M32:I28R_M32_pde:--model-nx 32")
best_rtps() {  # best RTPS on seed 0 over all c = 16 runs of a case (reads JSONs)
  $PY - "$1" <<'PYEOF'
import glob, json, re, sys
base = sys.argv[1]; best = None
for f in glob.glob(f"artifacts/experiments/da2d/i11_{base}_rtps*.json"):
    m = re.search(r"_rtps([0-9.]+)\.json$", f)
    if not m: continue
    e = json.load(open(f))["steady"]["analysis"]
    if best is None or e < best[0]: best = (e, m.group(1))
print(best[1] if best else "0.7")
PYEOF
}
lane=$1
for i in "${!CASES[@]}"; do
  [ $((i % 2)) -eq $lane ] || continue
  IFS=: read name base args <<< "${CASES[$i]}"
  # J2: RTPS refinement (c = 16)
  for r in 0.6 0.8; do pde ${base}_rtps$r --loc-c 16 $args --rtps $r; done
  case $name in adv*) pde ${base}_rtps1.0 --loc-c 16 $args --rtps 1.0 ;; esac
  b=$(best_rtps $base)
  # J2: localization and inflation at the best RTPS / as an alternative
  for c in 8.25 24; do pde J2_${name}_c${c}_rtps$b --loc-c $c $args --rtps $b; done
  for inf in 1.1 1.2; do pde J2_${name}_inf$inf --loc-c 16 $args --rtps 0 --inflation $inf; done
  # J1: seeds 1-2 at the best RTPS (c = 16)
  for sd in 1 2; do pde J1_${name}_rtps${b}_s$sd --loc-c 16 $args --rtps $b --seed $sd; done
done
echo "$(date '+%F %T') LANE $lane DONE" >> $LOG
