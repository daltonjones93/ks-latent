#!/bin/bash
# J17S screening (2026-10-10): candidate 1 (ganiso) and 2 (aniso) truths -- statistics, KS-model and best-fit
# one-cycle errors (exp_j17s_screen.py) and truth Lyapunov spectra (j17s_lyap.py), 10 jobs at a time, CPU only.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/j17s.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PY LOG
job() { kind=$1; lev=$2; what=$3
  if [ $what = screen ]; then f=artifacts/experiments/da2d/j17s_screen_$kind$lev.json; cmd="scripts/exp_j17s_screen.py"
  else f=artifacts/experiments/da2d/j17s_lyap_$kind$lev.json; cmd="scripts/j17s_lyap.py"; fi
  [ -f $f ] && return
  $PY $cmd --kind $kind --level $lev > artifacts/logs/j17s_${what}_$kind$lev.log 2>&1
  echo "$(date '+%F %T') done $what $kind$lev rc=$? $(tail -1 artifacts/logs/j17s_${what}_$kind$lev.log | cut -c1-300)" >> $LOG; }
export -f job
{ for l in 0 0.05 0.1 0.2 0.3 0.5 0.7; do echo "ganiso $l screen"; done
  for l in 0.05 0.1 0.25 0.5 1; do echo "aniso $l screen"; done
  for l in 0.1 0.2 0.3 0.5 0.7 0.05; do echo "ganiso $l lyap"; done
  for l in 0.1 0.25 0.5 1 0.05; do echo "aniso $l lyap"; done; } | xargs -P 10 -L 1 bash -c 'job $0 $1 $2'
echo "$(date '+%F %T') J17S ALL DONE" >> $LOG
