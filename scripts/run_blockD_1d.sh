#!/bin/bash
# Block D, 1D data assimilation (docs/full_paper_tasks.md D1 and the 1D half of D2), 2026-10-08.
# Model: Section 224 (the paper's 1D model: stage1 ..._d48_localmlp + stage2 ..._warmstart_k12_300ep).
# All runs: 100 cycles x 5 steps, fixed physical observation spacing (--obs-spacing-units),
# per-step arrays and steady-state (second-half) skill in each JSON (artifacts/da_pff_ltransfer_<tag>.json).
# Usage: run_blockD_1d.sh {d1_pff_200|d1_pff_400|d1_letkf_200|d1_letkf_400|d2|d3}
# d3 (task D3): SEC at L=200, fixed spacing: raw SEC + PFF, PSD-projected SEC + PFF, serial SEC EnKF;
#     outputs artifacts/e3_sec_vs_gc_ltransfer_d3_*.json (GC comparators are the D1 runs).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=3
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
AE=artifacts/stage1_ae_patched_full_section224_ks_localfield_d48_localmlp.pt
PR=artifacts/stage2_prop_patched_full_section224_ks_localfield_d48_localmlp_warmstart_k12_300ep.pt
LOG=artifacts/logs/blockD_1d.log; OUT=artifacts/experiments/d1
mkdir -p $OUT artifacts/logs
log () { echo "[blockD-1d $1 $(date '+%H:%M:%S')] $2" >> $LOG; }
sweep () {  # method L
  log "$1_$2" "start"
  $PY scripts/exp_e2_ensemble_sweep_ltransfer.py --ae-checkpoint $AE --prop-checkpoint $PR --new-L $2 \
    --n-cycles 100 --method $1 --obs-spacing-units 3.125 --localizers none gaspari_cohn \
    --tag-prefix d1_$1_L$2 --out-csv $OUT/$( [ $1 = pff ] && echo e2b || echo e7b )_L$2.csv \
    > artifacts/logs/blockD_1d_$1_L$2.log 2>&1 && log "$1_$2" "done" || log "$1_$2" "FAILED"
}
case $1 in
  d1_pff_200) sweep pff 200 ;; d1_pff_400) sweep pff 400 ;;
  d1_letkf_200) sweep letkf 200 ;; d1_letkf_400) sweep letkf 400 ;;
  d2)
    for sp in 1.5625 6.25 12.5; do  # 3.125 is the D1 N=64 GC point
      tag=d2_pff_L200_gc_N64_sp$sp
      log d2 "start $tag"
      $PY scripts/run_da_pff_ltransfer.py --ae-checkpoint $AE --prop-checkpoint $PR --new-L 200 --n-ensemble 64 \
        --n-cycles 100 --method pff --localizer gaspari_cohn --obs-spacing-units $sp --tag $tag \
        > artifacts/logs/blockD_1d_$tag.log 2>&1 && log d2 "done $tag" || log d2 "FAILED $tag"
    done ;;
  d3)
    for N in 8 16 32 64 128 256; do
      for v in "sec pff" "sec_psd pff" "none serial_sec_enkf"; do
        set -- $v; loc=$1; meth=$2
        tag=d3_L200_${meth}_${loc}_N$N
        log d3 "start $tag"
        $PY scripts/exp_e3_sec_vs_gc_at_transfer.py --ae-checkpoint $AE --prop-checkpoint $PR --new-L 200 \
          --n-ensemble $N --n-cycles 100 --obs-spacing-units 3.125 --localizer $loc --method $meth --tag $tag \
          > artifacts/logs/blockD_1d_$tag.log 2>&1 && log d3 "done $tag" || log d3 "FAILED $tag"
      done
    done ;;
  *) echo "usage: $0 {d1_pff_200|d1_pff_400|d1_letkf_200|d1_letkf_400|d2|d3}"; exit 1 ;;
esac
