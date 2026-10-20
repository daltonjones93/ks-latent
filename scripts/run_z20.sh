#!/bin/bash
# Z20 (2026-10-20, docs/split_paper_tasks.md): nonlinearity of the latent coordinates, Paper II.
# Serial queue, one job at a time, 2 threads, efficiency cores. Order: E1 -> E3 -> E4 -> E2 -> E5 (-> E6, separate).
# Model p7c12d10_ks2d, 44^2, paper settings (K = 4 tilings x 64 members, RTPS 0.7, GC 1.5 sites, stride 8,
# sigma_o 0.1, 5 snapshots per cycle, 100 cycles, steady state = cycles 51-100). Skips runs whose json exists.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
LOG=artifacts/logs/z20.log; E=artifacts/experiments/da2d; TAG=p7c12d10_ks2d
mkdir -p artifacts/logs
log() { echo "$(date '+%F %T') [Z20] $*" >> $LOG; }
lat() { out=$1; shift; [ -f $E/h9_${out}.json ] && return
  taskpolicy -b $PY scripts/exp_h9_tiled_da.py --tag $TAG --rtps 0.7 --no-free --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
  log "done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-260 | tr '\n' ' ')"; }
phys() { out=$1; shift; [ -f $E/i11_${out}.json ] && return
  taskpolicy -b $PY scripts/exp_i11_pde_letkf.py --workers 2 --out $out "$@" > artifacts/logs/i11_$out.log 2>&1
  log "done i11 $out rc=$? $(grep -E "^$out " artifacts/logs/i11_$out.log | cut -c1-200)"; }
best() {  # best() prefix key cand1 cand2 ... -> prints the candidate with the lowest steady[key]
  $PY - "$@" <<'PY'
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
log "start"
# ---------------- E1: diagnostics at the paper configuration ----------------
for s in 0 1 2; do lat z20_e1_s$s --nl-diag --seed $s; done
log "E1 DONE"
# ---------------- E3: iterative per-site update ----------------
for n in 2 3; do for s in 0 1 2; do lat z20_e3_it${n}_s$s --n-iter $n --iter-tol 0 --seed $s; done; done
for r in 0.5 0.8 0.9; do lat z20_e3_it3_r${r}_s0 --n-iter 3 --iter-tol 0 --rtps $r; done
log "E3 DONE"
# ---------------- E4: oracle forecast vs perfect-model physics LETKF ----------------
for mode in increment project; do for r in 0.5 0.7 0.9; do lat z20_e4_${mode}_r${r}_s0 --oracle-forecast $mode --rtps $r --nl-diag; done; done
for mode in increment project; do
  rb=$(best h9_z20_e4_${mode}_r blend 0.5_s0 0.7_s0 0.9_s0); rb=${rb%_s0}; log "E4 $mode best RTPS $rb"
  for s in 1 2; do lat z20_e4_${mode}_r${rb}_s$s --oracle-forecast $mode --rtps $rb --nl-diag --seed $s; done
  for s in 0 1 2; do lat z20_e4_${mode}_r${rb}_N16_s$s --oracle-forecast $mode --rtps $rb --n-ensemble 16 --seed $s; done
done
rb=$(best h9_z20_e4_increment_r blend 0.5_s0 0.7_s0 0.9_s0); rb=${rb%_s0}
for s in 0 1 2; do lat z20_e4_increment_r${rb}_it3_s$s --oracle-forecast increment --rtps $rb --n-iter 3 --iter-tol 0 --seed $s; done
# perfect-model physics LETKF on the same truth and observations (seed-0 tuning grid exists as i11_L44_st8_*; best c24 rtps0)
for s in 0 1 2; do phys z20_e4_phys_c24_r0_s$s --loc-c 24 --rtps 0 --seed $s; done
for s in 0 1 2; do phys z20_e4_phys_c24_r0_N256_s$s --loc-c 24 --rtps 0 --n-ensemble 256 --seed $s; done
log "E4 DONE"
# ---------------- E2: Gaussianity with 256 members per tiling ----------------
lat z20_e2_N256_s0 --n-ensemble 256 --save-priors 10 --nl-diag
taskpolicy -b $PY scripts/exp_z20_gaussianity.py --priors $E/h9_z20_e2_N256_s0_priors.npz --out z20_e2 >> $LOG 2>&1
log "E2 DONE"
# ---------------- E5: observation error x cycle length (seed 0) ----------------
for so in 0.05 0.1 0.2 0.4; do for np_ in 5 10 20; do
  cell="so${so}_p${np_}"
  for r in 0.5 0.7 0.9; do lat z20_e5_std_${cell}_r${r} --obs-noise-std $so --n-prop-steps $np_ --rtps $r --nl-diag; done
  rb=$(best h9_z20_e5_std_${cell}_r blend 0.5 0.7 0.9); log "E5 $cell best RTPS $rb"
  lat z20_e5_it3_${cell} --obs-noise-std $so --n-prop-steps $np_ --rtps $rb --n-iter 3 --iter-tol 0
  lat z20_e5_orc_${cell} --obs-noise-std $so --n-prop-steps $np_ --rtps $rb --oracle-forecast increment --nl-diag
  for r in 0 0.3; do phys z20_e5_phys_${cell}_r${r} --obs-noise-std $so --n-prop-steps $np_ --loc-c 24 --rtps $r; done
done; done
log "E5 DONE"
log "Z20 DONE"
