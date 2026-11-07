#!/bin/bash
# Track FP, FP0: known-answer pilot for a transport + sources law in hand-built CM1 fields (CPU).
# Runs the unit tests, a short smoke on two storms, then the full pilot.
#   cd ~/latent_DA && sbatch cluster/cashew_gpu/f0_pilot.slurm
#   variables: OUT (default ~/cm1_da/f0), RUNS (raw run dirs), FIT (how many of RUNS fit the laws),
#   EXTRA (more fp_pilot.py options, e.g. "--dt-stride 6 --t-range 60 180")
#SBATCH --job-name=cm1f0
#SBATCH --partition=coe_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cm1f0_%j.out
set -euo pipefail
cd "$HOME/latent_DA"
. "$HOME/venvs/cm1/bin/activate"
export PYTHONPATH=$HOME/latent_DA OMP_NUM_THREADS=8
B=$HOME/cm1_bundle
RUNS=${RUNS:-"$B/ens_A/members/mem001 $B/ens_A/members/mem002 $B/ens_A/members/mem003 $B/ens_A/members/mem004 $B/ens_A2/members/mem001 $B/ens_A2/members/mem002 $B/ens_B/members/mem003 $B/ens_B/members/mem016"}
set -- $RUNS
python -m pytest -q tests/unit/test_cm1_fieldpde.py -p no:cacheprovider
python scripts/cm1/fp_pilot.py --runs "$1" "$2" --fit 1 --out "$HOME/cm1_da/f0_smoke_$SLURM_JOB_ID" \
    --t-range 90 120 --t-step 15 --n-points 2000 ${EXTRA:-} | tail -12
python scripts/cm1/fp_pilot.py --runs "$@" --fit "${FIT:-4}" --out "${OUT:-$HOME/cm1_da/f0}" ${EXTRA:-}
