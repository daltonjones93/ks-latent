#!/bin/bash
# run_z20e.sh: run_z20d.sh with the projected oracle variant retuned (2026-10-20 17:55).
# run_z20d.sh (user 2026-10-20 17:30: "drop the 256-member physics seeds 1-2; do whatever is fastest"):
# one serial lane, 4 threads, performance cores (two parallel lanes on efficiency cores ran each oracle job ~3x slower).
# Order: rest of E4 oracle -> E2 -> E5 (5-snapshot cells incl. oracle first, then 10 and 20) -> "Z20 DONE".
# run_z20c.sh: as run_z20b.sh with the oracle RTPS grid {0, 0.3, 0.5, 0.7} and the E5 oracle cells at the best
# E4 oracle RTPS (oracle has no model noise; the standard filter's RTPS does not transfer).
# Z20 trimmed queue (user 2026-10-20 "get results ASAP, trim both"), replaces the rest of run_z20.sh:
# E4: projected variant at RTPS 0.7 only, no 16-member oracle runs, no relinearized oracle runs;
# E5: oracle runs at the 5-snapshot cycle only. Two lanes at 2 threads each (oracle runs | everything else).
# Writes "Z20 DONE" when both lanes finish (run_z20_e6.sh waits for it). Skips runs whose json exists.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
LOG=artifacts/logs/z20.log; E=artifacts/experiments/da2d; TAG=p7c12d10_ks2d
log() { echo "$(date '+%F %T') [Z20] $*" >> $LOG; }
lat() { out=$1; shift; [ -f $E/h9_${out}.json ] && return
  $PY scripts/exp_h9_tiled_da.py --tag $TAG --rtps 0.7 --no-free --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
  log "done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-260 | tr '\n' ' ')"; }
phys() { out=$1; shift; [ -f $E/i11_${out}.json ] && return
  $PY scripts/exp_i11_pde_letkf.py --workers 4 --out $out "$@" > artifacts/logs/i11_$out.log 2>&1
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

log "queue run_z20e.sh start (serial; replaces run_z20d.sh: projected oracle variant retuned, it collapsed at RTPS 0.3)"
while pgrep -f "out z20_e4_increment_r0.3_s1" > /dev/null; do sleep 60; done  # left running by run_z20d
[ -f $E/h9_z20_e4_increment_r0.3_s1.json ] && log "done z20_e4_increment_r0.3_s1 (from run_z20d.sh) $($PY -c "import json; s=json.load(open('$E/h9_z20_e4_increment_r0.3_s1.json'))['steady']; print(round(s['blend'],3), 'calib', round(s['calib_blend'],2))")"
rb=0.3  # best oracle-incremental RTPS (seed 0: 0 0.192, 0.3 0.152, 0.5 0.174, 0.7 0.246)
lat z20_e4_increment_r${rb}_s2 --oracle-forecast increment --rtps $rb --nl-diag --seed 2
# projected variant: members return to the decoder's range each cycle, so with no model noise the spread decays
# (RTPS 0.3: spread/error 0.17, error 0.679). Tune RTPS 0.9, and latent noise 0.045 (= the learned filter's 0.02 per
# step over 5 steps) and 0.1 at RTPS 0.7.
lat z20_e4_project_r0.9_s0 --oracle-forecast project --rtps 0.9 --nl-diag
lat z20_e4_project_r0.7_n0.045_s0 --oracle-forecast project --rtps 0.7 --oracle-latent-noise 0.045 --nl-diag
lat z20_e4_project_r0.7_n0.1_s0 --oracle-forecast project --rtps 0.7 --oracle-latent-noise 0.1 --nl-diag
pb=$(best h9_z20_e4_project_r blend 0.3_s0 0.9_s0 0.7_n0.045_s0 0.7_n0.1_s0); pb=${pb%_s0}; log "E4 project best $pb"
pr=${pb%%_n*}; pn=0; [[ $pb == *_n* ]] && pn=${pb##*_n}
for s in 1 2; do lat z20_e4_project_r${pb}_s$s --oracle-forecast project --rtps $pr --oracle-latent-noise $pn --nl-diag --seed $s; done
log "E4 DONE"
# ---- E2 ----
lat z20_e2_N256_s0 --n-ensemble 256 --save-priors 10 --nl-diag
$PY scripts/exp_z20_gaussianity.py --priors $E/h9_z20_e2_N256_s0_priors.npz --out z20_e2 >> $LOG 2>&1
log "E2 DONE"
# ---- E5 ----
for np_ in 5 10 20; do for so in 0.05 0.1 0.2 0.4; do
  cell="so${so}_p${np_}"
  for r in 0.5 0.7 0.9; do lat z20_e5_std_${cell}_r${r} --obs-noise-std $so --n-prop-steps $np_ --rtps $r --nl-diag; done
  rs=$(best h9_z20_e5_std_${cell}_r blend 0.5 0.7 0.9); log "E5 $cell best RTPS $rs"
  lat z20_e5_it3_${cell} --obs-noise-std $so --n-prop-steps $np_ --rtps $rs --n-iter 3 --iter-tol 0
  for r in 0 0.3; do phys z20_e5_phys_${cell}_r${r} --obs-noise-std $so --n-prop-steps $np_ --loc-c 24 --rtps $r; done
  [ $np_ = 5 ] && lat z20_e5_orc_${cell} --obs-noise-std $so --n-prop-steps 5 --rtps $rb --oracle-forecast increment --nl-diag
done; done
log "E5 DONE"
log "Z20 DONE"
