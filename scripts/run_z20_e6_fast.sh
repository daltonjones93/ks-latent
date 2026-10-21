#!/bin/bash
# Z20 E6 (2026-10-20): kernel particle flow filter (Hu & van Leeuwen 2021) on the 2D latent field, 44^2.
# Starts after run_z20.sh finishes (waits for "Z20 DONE"); same conventions (serial, 2 threads, efficiency cores).
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
LOG=artifacts/logs/z20.log; E=artifacts/experiments/da2d; TAG=p7c12d10_ks2d
log() { echo "$(date '+%F %T') [Z20] $*" >> $LOG; }
lat() { out=$1; shift; [ -f $E/h9_${out}.json ] && return
  $PY scripts/exp_h9_tiled_da.py --tag $TAG --no-free --filter kpff --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
  log "done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-260 | tr '\n' ' ')"; }
best() { $PY - "$@" <<'PY'
import json, sys
pre, key, cands = sys.argv[1], sys.argv[2], sys.argv[3:]
v = {}
for c in cands:
    try:
        s = json.load(open(f"artifacts/experiments/da2d/{pre}{c}.json"))["steady"]
        if s.get("diverged_at_cycle") is None:
            v[c] = s[key]
    except FileNotFoundError:
        pass
print(min(v, key=v.get) if v else cands[0])
PY
}
until grep -q "Z20 DONE" $LOG 2>/dev/null; do sleep 600; done
log "E6 start"
for N in 32 16; do
  for al in def 0.1 0.5; do for r in 0 0.7; do
    aa=""; [ $al != def ] && aa="--kpff-alpha $al"
    lat z20_e6_N${N}_a${al}_r${r}_s0 --n-ensemble $N --rtps $r $aa --nl-diag
  done; done
  b=$(best h9_z20_e6_N${N}_a blend def_r0_s0 def_r0.7_s0 0.1_r0_s0 0.1_r0.7_s0 0.5_r0_s0 0.5_r0.7_s0); b=${b%_s0}
  al=${b%%_r*}; r=${b##*_r}; aa=""; [ $al != def ] && aa="--kpff-alpha $al"
  log "E6 N$N best alpha $al RTPS $r"
  for s in 1 2; do lat z20_e6_N${N}_a${al}_r${r}_s$s --n-ensemble $N --rtps $r $aa --seed $s --nl-diag; done
done
# the E5 cell with the largest nonlinearity index (standard filter, best RTPS), kernel PFF N32 at its best setting
cell=$($PY - <<'PY'
import glob, json
best = None
for f in glob.glob("artifacts/experiments/da2d/h9_z20_e5_std_*.json"):
    s = json.load(open(f))["steady"]
    if s.get("nl_eta_o") is not None and (best is None or s["nl_eta_o"] > best[0]):
        best = (s["nl_eta_o"], f.split("h9_z20_e5_std_")[1].rsplit("_r", 1)[0], json.load(open(f))["args"])
print(best[1], best[2]["obs_noise_std"], best[2]["n_prop_steps"])
PY
)
read cname so np_ <<< "$cell"
b=$(best h9_z20_e6_N32_a blend def_r0_s0 def_r0.7_s0 0.1_r0_s0 0.1_r0.7_s0 0.5_r0_s0 0.5_r0.7_s0); b=${b%_s0}
al=${b%%_r*}; r=${b##*_r}; aa=""; [ $al != def ] && aa="--kpff-alpha $al"
lat z20_e6_e5cell_${cname} --n-ensemble 32 --rtps $r $aa --obs-noise-std $so --n-prop-steps $np_ --nl-diag
log "E6 DONE"
