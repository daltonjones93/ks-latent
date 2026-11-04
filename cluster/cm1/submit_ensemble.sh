#!/bin/bash
# Submit all unfinished members of an ensemble as a few packed Slurm jobs.
# Cashew caps each partition at 20 submitted+pending jobs, so members are packed
# CPUS per job instead of one job each. Safe to re-run: finished members are skipped.
# Usage: bash submit_ensemble.sh ENS_DIR [CPUS_PER_JOB=8] [extra sbatch options...]
#   e.g. bash kit/submit_ensemble.sh ens_A 8 --mail-type=END --mail-user=you@colostate.edu
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
ENS=$(cd "$1" && pwd); CPUS=${2:-8}; shift; [ $# -gt 0 ] && shift
todo=()
for d in "$ENS"/members/mem*; do [ -f "$d/cm1.done" ] || todo+=("$(basename "$d")"); done
n=${#todo[@]}
[ "$n" -eq 0 ] && { echo "nothing to do: all members done"; exit 0; }
njobs=$(( (n + CPUS - 1) / CPUS ))
queued=$(squeue -h -u "$USER" 2>/dev/null | wc -l | tr -d ' ')
if [ $((njobs + queued)) -gt 18 ]; then
  echo "would make $njobs jobs with $queued already queued (cap is 20): use more CPUs per job"; exit 1
fi
mkdir -p logs
echo "$n members -> $njobs jobs of up to $CPUS CPUs"
for ((j = 0; j < njobs; j++)); do
  chunk=("${todo[@]:$((j * CPUS)):$CPUS}")
  sbatch --cpus-per-task="${#chunk[@]}" "$@" "$KIT/cm1_pack.slurm" "$KIT" "$ENS" "${chunk[@]}"
done
