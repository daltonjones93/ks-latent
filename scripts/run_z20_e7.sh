#!/bin/bash
# Z20 E7 (2026-10-20): 1D data assimilation rerun with the kernel particle flow filter (Hu & van Leeuwen 2021),
# same model, truth, observations and configurations as the D1 runs of Paper II App. D.3 (scripts/run_blockD_1d.sh):
# L in {200, 400}, localizer none / Gaspari-Cohn (half-width 3 sites), N in {8..256}, spacing 3.125, 100 cycles.
# Starts after run_z20_e6.sh (waits for "E6 DONE"). Serial, 2 threads, efficiency cores.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
LOG=artifacts/logs/z20.log
AE=artifacts/stage1_ae_patched_full_section224_ks_localfield_d48_localmlp.pt
PR=artifacts/stage2_prop_patched_full_section224_ks_localfield_d48_localmlp_warmstart_k12_300ep.pt
log() { echo "$(date '+%F %T') [Z20] $*" >> $LOG; }
run() { tag=$1; shift; [ -f artifacts/da_pff_ltransfer_$tag.json ] && return
  taskpolicy -b $PY scripts/run_da_pff_ltransfer.py --ae-checkpoint $AE --prop-checkpoint $PR --n-cycles 100 \
    --method kpff --obs-spacing-units 3.125 --tag $tag "$@" > artifacts/logs/z20_$tag.log 2>&1
  log "done $tag rc=$? $($PY -c "import json; d=json.load(open('artifacts/da_pff_ltransfer_$tag.json')); print('skill', round(d['skill_steady'],2), 'calib', round(d['calibration_steady'],2))" 2>/dev/null)"; }
until grep -q "E6 DONE" $LOG 2>/dev/null; do sleep 600; done
log "E7 start"
for al in def 0.1 0.5; do for r in 0 0.7; do
  aa=""; [ $al != def ] && aa="--kpff-alpha $al"
  run z20_e7_tune_a${al}_r${r} --new-L 200 --n-ensemble 32 --localizer gaspari_cohn --kpff-rtps $r $aa
done; done
b=$($PY - <<'PY'
import json
v = {}
for al in ("def", "0.1", "0.5"):
    for r in ("0", "0.7"):
        try:
            v[(al, r)] = json.load(open(f"artifacts/da_pff_ltransfer_z20_e7_tune_a{al}_r{r}.json"))["skill_steady"]
        except FileNotFoundError:
            pass
al, r = max(v, key=v.get)
print(al, r)
PY
)
read al r <<< "$b"; aa=""; [ $al != def ] && aa="--kpff-alpha $al"
log "E7 best alpha $al RTPS $r (max steady skill at L=200, N=32, GC)"
for L in 200 400; do for loc in gaspari_cohn none; do for N in 8 16 32 64 128 256; do
  run z20_e7_kpff_L${L}_${loc}_N$N --new-L $L --n-ensemble $N --localizer $loc --kpff-rtps $r $aa
done; done; done
log "E7 DONE"
