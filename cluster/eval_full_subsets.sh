#!/bin/bash
# Re-score the full-run models (cluster/saidie/full_run.sh) on each ensemble's test storms separately, with
# true and held boundaries (H0), so they compare with the ens_A and ens_B screen numbers. CPU or GPU:
#   DEVICE=cpu bash cluster/eval_full_subsets.sh            (on a compute node; Cashew: via sbatch --wrap)
# Writes results_<subset>_<boundary>.json in each run directory and prints one table per subset.
set -euo pipefail
cd "$HOME/latent_DA"
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$HOME/latent_DA
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
RUNS=${RUNS:-"full_v1 full_v1_stagec full_v1_crps full_v1_stagec_crps"}
for r in $RUNS; do
  for sub in ensA:mem ensA2:${A2} ensB:${B}; do
    for b in truth persistence; do
      echo "=== $r ${sub%%:*} $b"
      python scripts/cm1/train_cm1.py --profile full --source "$DATA" --device "${DEVICE:-cuda}" --threads "${THREADS:-16}" \
          --out "$HOME/cm1_runs/$r" --eval-only --val-members "$VAL" --test-members "$TEST" --eval-samples 8 \
          --eval-subset "${sub#*:}" --eval-boundary $b --results-tag "${sub%%:*}_$b" 2>&1 | grep -E "Error|evaluating on" || true
    done
  done
done
for sub in ensA ensA2 ensB; do
  for b in truth persistence; do
    echo; echo "##### $sub, boundary $b"
    python scripts/cm1/compare_runs.py $(for r in $RUNS; do f=$HOME/cm1_runs/$r/results_${sub}_$b.json; [ -f "$f" ] && echo "$f"; done)
  done
done
