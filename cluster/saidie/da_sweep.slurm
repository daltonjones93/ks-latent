#!/bin/bash -l
# A list of DA studies under the G0 protocol (validation tuning, then test storms): scripts/cm1/da_sweep.py.
#   cd ~/latent_DA && sbatch --export=ALL,SPEC=cluster/saidie/specs/n5_loc.json,ROOT=fable cluster/saidie/da_sweep.slurm
#SBATCH --job-name=cm1sweep
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cm1sweep_%j.out
set -euo pipefail
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES not set: refusing to run on an unassigned GPU"; exit 1; }
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8
cd "$REPO"
python -m pytest -q tests/unit/test_cm1*.py -p no:cacheprovider
python scripts/cm1/da_sweep.py --spec "${SPEC:?set SPEC}" --root "$HOME/cm1_da/${ROOT:-fable}" \
    --source "cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B" ${ONLY:+--only $ONLY}
