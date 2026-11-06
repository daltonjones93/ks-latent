#!/bin/bash
# H2: where forecast skill runs out. The full-run models scored to 2 h lead on all 22 test storms, boundaries
# held at the start state; start times every 30 min so that t0 + 2 h fits in the 6 h runs.
#   DEVICE=cpu bash cluster/eval_h2.sh          (Cashew: via sbatch --wrap)
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
for r in ${RUNS:-full_v1_stagec full_v1_stagec_crps}; do
  echo "=== $r"
  python scripts/cm1/train_cm1.py --profile full --source "$DATA" --device "${DEVICE:-cuda}" --threads "${THREADS:-16}" \
      --out "$HOME/cm1_runs/$r" --eval-only --val-members "$VAL" --test-members "$TEST" --eval-samples 8 \
      --eval-lead-min 120 --eval-every-min 30 --eval-boundary persistence --results-tag h2_lead120 2>&1 | grep -E "Error|^ +[0-9.]+ +[0-9]" || true
done
