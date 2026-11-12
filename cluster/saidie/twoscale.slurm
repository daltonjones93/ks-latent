#!/bin/bash -l
# 6a: train the small-scale half of the two-scale model on a frozen large run. One GPU.
#   sbatch --export=ALL,OUT=ts_v1,EXTRA="" cluster/saidie/twoscale.slurm
#SBATCH --job-name=twoscale
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/twoscale_%j.out
set -euo pipefail
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES not set: refusing to run on an unassigned GPU"; exit 1; }
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=16
cd "$REPO"
python -m pytest -q tests/unit/test_cm1_twoscale.py -p no:cacheprovider
python scripts/cm1/train_twoscale.py --large-run "$HOME/cm1_runs/${LARGE:-fpc_nested_p8_dft}" \
    --source "cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B" --out "$HOME/cm1_runs/${OUT:?}" --device cuda ${EXTRA:-}
