#!/bin/bash
# J16: physics filter with standard model-error remedies. Waits for J1/J2 (run_j12.sh) to finish so each case
# uses J2's best RTPS. Lowest priority (nice 19, 2 threads). Seeds 0-2, stride 8, 100 cycles, c = 16.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/j16.log
until [ "$(grep -c 'LANE [01] DONE' artifacts/logs/j12.log 2>/dev/null)" -ge 2 ]; do sleep 120; done
# 12 J2 runs (forcing cases) crashed at 21:13-21:19 from a since-fixed print bug; the J2 lanes skip finished
# runs, so rerunning them fills exactly the gaps (and J1 seeds then use the completed best RTPS).
echo "$(date '+%F %T') J1/J2 pass 1 done; rerunning J2 lanes to fill crashed runs" >> $LOG
scripts/run_j12.sh 0 & scripts/run_j12.sh 1 & wait
echo "$(date '+%F %T') J1/J2 complete; starting J16" >> $LOG
ENV="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2"
pde() { tag=$1; shift
  [ -f artifacts/experiments/da2d/i11_${tag}.json ] && return
  env $ENV nice -n 19 $PY scripts/exp_i11_pde_letkf.py --workers 2 --loc-c 16 --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/i11_${tag}.log | cut -c1-170)" >> $LOG; }
best_rtps() { $PY - "$1" <<'PYEOF'
import glob, json, re, sys
best = None
for f in glob.glob(f"artifacts/experiments/da2d/i11_{sys.argv[1]}_rtps*.json"):
    m = re.search(r"_rtps([0-9.]+)\.json$", f)
    if not m: continue
    e = json.load(open(f))["steady"]["analysis"]
    if best is None or e < best[0]: best = (e, m.group(1))
print(best[1] if best else "0.7")
PYEOF
}
# model-error statistics (independent seed-3 truth) for the bias / additive-noise cases
for spec in "frc0.2:--model-forcing-amp 0.2" "frc0.4:--model-forcing-amp 0.4" "M32:--model-nx 32"; do
  name=${spec%%:*}; args=${spec#*:}
  [ -f artifacts/experiments/da2d/j16_model_error_${name}.npz ] || \
    env $ENV nice -n 19 $PY scripts/exp_j16_estimate_pde_model_error.py $args --n-starts 100 --out $name >> $LOG 2>&1
done
# (a) online parameter estimation
for spec in "lap1.1:I28W_lap1.1:--model-lap-coef 1.1 --estimate-param lap --param-init-std 0.05" \
            "lap1.25:I28W_lap1.25:--model-lap-coef 1.25 --estimate-param lap --param-init-std 0.05" \
            "adv1:I28W_adv1:--model-advect-x 1 --estimate-param adv --param-init-std 0.3" \
            "adv2:I28W_adv2:--model-advect-x 2 --estimate-param adv --param-init-std 0.3"; do
  IFS=: read name base args <<< "$spec"; r=$(best_rtps $base)
  for sd in 0 1 2; do pde J16a_${name}_rtps${r}_s$sd $args --rtps $r --seed $sd; done
done
# (b) bias correction +- additive model-error noise; seed 0 sweep, then seeds 1-2 at the best q
for spec in "frc0.2:I28W_frc0.2:--model-forcing-amp 0.2" "frc0.4:I28W_frc0.4:--model-forcing-amp 0.4" "M32:I28R_M32_pde:--model-nx 32"; do
  IFS=: read name base args <<< "$spec"; r=$(best_rtps $base); f=artifacts/experiments/da2d/j16_model_error_${name}.npz
  for q in 0 0.5 1 2; do pde J16b_${name}_rtps${r}_bc_q${q}_s0 $args --rtps $r --model-error-file $f --bias-correct --q-scale $q; done
  bq=$($PY - "$name" "$r" <<'PYEOF'
import glob, json, re, sys
best = None
for f in glob.glob(f"artifacts/experiments/da2d/i11_J16b_{sys.argv[1]}_rtps{sys.argv[2]}_bc_q*_s0.json"):
    q = re.search(r"_q([0-9.]+)_s0\.json$", f).group(1); e = json.load(open(f))["steady"]["analysis"]
    if best is None or e < best[0]: best = (e, q)
print(best[1] if best else "0")
PYEOF
)
  for sd in 1 2; do pde J16b_${name}_rtps${r}_bc_q${bq}_s$sd $args --rtps $r --model-error-file $f --bias-correct --q-scale $bq --seed $sd; done
done
echo "$(date '+%F %T') J16 DONE" >> $LOG
