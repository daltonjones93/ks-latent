#!/bin/bash
# Z20 trimmed queue (user 2026-10-20 "get results ASAP, trim both"), replaces the rest of run_z20.sh:
# E4: projected variant at RTPS 0.7 only, no 16-member oracle runs, no relinearized oracle runs;
# E5: oracle runs at the 5-snapshot cycle only. Two lanes at 2 threads each (oracle runs | everything else).
# Writes "Z20 DONE" when both lanes finish (run_z20_e6.sh waits for it). Skips runs whose json exists.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
LOG=artifacts/logs/z20.log; E=artifacts/experiments/da2d; TAG=p7c12d10_ks2d
log() { echo "$(date '+%F %T') [Z20] $*" >> $LOG; }
lat() { out=$1; shift; [ -f $E/h9_${out}.json ] && return
  taskpolicy -b $PY scripts/exp_h9_tiled_da.py --tag $TAG --rtps 0.7 --no-free --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
  log "done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-260 | tr '\n' ' ')"; }
phys() { out=$1; shift; [ -f $E/i11_${out}.json ] && return
  taskpolicy -b $PY scripts/exp_i11_pde_letkf.py --workers 2 --out $out "$@" > artifacts/logs/i11_$out.log 2>&1
  log "done i11 $out rc=$? $(grep -E "^$out " artifacts/logs/i11_$out.log | cut -c1-200)"; }
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
laneA() {  # oracle-forecast runs
  while pgrep -f "out z20_e4_increment_r0.7_s0" > /dev/null; do sleep 60; done  # left running by run_z20.sh
  [ -f $E/h9_z20_e4_increment_r0.7_s0.json ] && log "done z20_e4_increment_r0.7_s0 (from run_z20.sh) $($PY -c "import json; print(round(json.load(open('$E/h9_z20_e4_increment_r0.7_s0.json'))['steady']['blend'],3))")"
  lat z20_e4_increment_r0.9_s0 --oracle-forecast increment --rtps 0.9 --nl-diag
  lat z20_e4_project_r0.7_s0 --oracle-forecast project --rtps 0.7 --nl-diag
  rb=$(best h9_z20_e4_increment_r blend 0.5_s0 0.7_s0 0.9_s0); rb=${rb%_s0}; log "E4 increment best RTPS $rb"
  for s in 1 2; do lat z20_e4_increment_r${rb}_s$s --oracle-forecast increment --rtps $rb --nl-diag --seed $s; done
  for s in 1 2; do lat z20_e4_project_r0.7_s$s --oracle-forecast project --rtps 0.7 --nl-diag --seed $s; done
  log "E4 oracle DONE"
  for so in 0.05 0.1 0.2 0.4; do
    cell="so${so}_p5"
    until [ -f $E/h9_z20_e5_std_${cell}_r0.9.json ]; do sleep 120; done
    rb=$(best h9_z20_e5_std_${cell}_r blend 0.5 0.7 0.9)
    lat z20_e5_orc_${cell} --obs-noise-std $so --n-prop-steps 5 --rtps $rb --oracle-forecast increment --nl-diag
  done
  log "laneA DONE"
}
laneB() {
  for s in 0 1 2; do phys z20_e4_phys_c24_r0_s$s --loc-c 24 --rtps 0 --seed $s; done
  for s in 0 1 2; do phys z20_e4_phys_c24_r0_N256_s$s --loc-c 24 --rtps 0 --n-ensemble 256 --seed $s; done
  log "E4 physics DONE"
  lat z20_e2_N256_s0 --n-ensemble 256 --save-priors 10 --nl-diag
  taskpolicy -b $PY scripts/exp_z20_gaussianity.py --priors $E/h9_z20_e2_N256_s0_priors.npz --out z20_e2 >> $LOG 2>&1
  log "E2 DONE"
  for np_ in 5 10 20; do for so in 0.05 0.1 0.2 0.4; do
    cell="so${so}_p${np_}"
    for r in 0.5 0.7 0.9; do lat z20_e5_std_${cell}_r${r} --obs-noise-std $so --n-prop-steps $np_ --rtps $r --nl-diag; done
    rb=$(best h9_z20_e5_std_${cell}_r blend 0.5 0.7 0.9); log "E5 $cell best RTPS $rb"
    lat z20_e5_it3_${cell} --obs-noise-std $so --n-prop-steps $np_ --rtps $rb --n-iter 3 --iter-tol 0
    for r in 0 0.3; do phys z20_e5_phys_${cell}_r${r} --obs-noise-std $so --n-prop-steps $np_ --loc-c 24 --rtps $r; done
  done; done
  log "laneB DONE"
}
log "trimmed queue run_z20b.sh start (two lanes)"
laneA & laneB & wait
log "E5 DONE"
log "Z20 DONE"
