#!/bin/bash
# Z20 follow-ups (2026-10-21 05:45), run after run_z20_e6_fast.sh ("E6 DONE"), before E7 (which now waits for
# "E6B DONE"): E6b kernel-PFF retuning of the model noise (the kernel PFF is over-dispersed at the EnKF's 0.02 per
# step: spread/error 1.27 at RTPS 0) and a 64-member comparison with the paper's EnKF; E5c sparser and denser
# observation networks (stride 16 and 4) with the nonlinearity diagnostics and the relinearized update.
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
until grep -q "E6 DONE" $LOG 2>/dev/null; do sleep 60; done
log "E6b/E5c start"
# ---- E6b: kernel PFF model noise (alpha = 1/N, RTPS 0: best of the E6 grid so far) ----
for mn in 0.01 0.005; do lat z20_e6b_N32_mn${mn}_s0 --filter kpff --n-ensemble 32 --rtps 0 --model-noise-std $mn --nl-diag; done
b=$(best h9_z20_e6b_N32_mn blend 0.01_s0 0.005_s0); b=${b%_s0}
e6=$($PY -c "import json; print(json.load(open('$E/h9_z20_e6_N32_adef_r0_s0.json'))['steady']['blend'])")
eb=$($PY -c "import json; print(json.load(open('$E/h9_z20_e6b_N32_mn${b}_s0.json'))['steady']['blend'])")
log "E6b best model noise $b ($eb vs $e6 at 0.02)"
mn=$($PY -c "print('$b' if $eb < $e6 else '0.02')")
for s in 1 2; do lat z20_e6b_N32_mn${mn}_s$s --filter kpff --n-ensemble 32 --rtps 0 --model-noise-std $mn --nl-diag --seed $s; done
for s in 0 1 2; do lat z20_e6b_N64_mn${mn}_s$s --filter kpff --n-ensemble 64 --rtps 0 --model-noise-std $mn --nl-diag --seed $s; done
# the EnKF at 32 members on seeds 1-2 (seed 0 exists: D10_L44_I3_N32, 0.330)
for s in 1 2; do lat z20_e6b_enkf_N32_s$s --n-ensemble 32 --rtps 0.7 --seed $s; done
log "E6B DONE"
# ---- E5c: observation density (seed 0, 5 snapshots, sigma_o 0.1) ----
for st in 16 4; do
  for r in 0.5 0.7 0.9; do lat z20_e5c_st${st}_r${r} --obs-stride $st --rtps $r --nl-diag; done
  rb=$(best h9_z20_e5c_st${st}_r blend 0.5 0.7 0.9); log "E5c stride $st best RTPS $rb"
  lat z20_e5c_it3_st${st} --obs-stride $st --rtps $rb --n-iter 3 --iter-tol 0
done
log "E5C DONE"
