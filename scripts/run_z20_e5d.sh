#!/bin/bash
# Z20 E5d (user 2026-10-21 "sure, add e5d"): the 20-snapshot E5 cells on three seeds, with RTPS 0.3 added
# (seed 0's best was 0.5, the grid edge). Runs after run_z20_after_e6.sh ("E5C DONE"); E7 waits for "E5D DONE".
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
LOG=artifacts/logs/z20.log; E=artifacts/experiments/da2d; TAG=p7c12d10_ks2d
log() { echo "$(date '+%F %T') [Z20] $*" >> $LOG; }
lat() { out=$1; shift; [ -f $E/h9_${out}.json ] && return
  $PY scripts/exp_h9_tiled_da.py --tag $TAG --no-free --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
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
until grep -q "E5C DONE" $LOG 2>/dev/null; do sleep 60; done
log "E5d start"
for so in 0.05 0.1 0.2 0.4; do
  cell="so${so}_p20"
  lat z20_e5_std_${cell}_r0.3 --obs-noise-std $so --n-prop-steps 20 --rtps 0.3 --nl-diag
  rb=$(best h9_z20_e5_std_${cell}_r blend 0.3 0.5 0.7 0.9); log "E5d $cell best RTPS $rb"
  [ "$rb" != "0.5" ] && lat z20_e5d_it3_${cell}_s0 --obs-noise-std $so --n-prop-steps 20 --rtps $rb --n-iter 3 --iter-tol 0
  for s in 1 2; do
    lat z20_e5d_std_${cell}_r${rb}_s$s --obs-noise-std $so --n-prop-steps 20 --rtps $rb --nl-diag --seed $s
    lat z20_e5d_it3_${cell}_r${rb}_s$s --obs-noise-std $so --n-prop-steps 20 --rtps $rb --n-iter 3 --iter-tol 0 --seed $s
  done
done
log "E5D DONE"
