#!/bin/bash -l
# E1b: train the composite-reflectivity operator for a trained run (saved into the run directory).
#   cd ~/latent_DA && sbatch --export=ALL,RUN=d34_c32_kuniform cluster/saidie/refl_head.slurm
#SBATCH --job-name=cm1refl
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cm1refl_%j.out
set -euo pipefail
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES not set: refusing to run on an unassigned GPU"; exit 1; }
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8
cd "$REPO"
python scripts/cm1/train_refl_head.py --run "$HOME/cm1_runs/${RUN:?set RUN}" --source "${DATA:-cache:$HOME/cm1_cache}" \
    --device cuda ${EXTRA:-}
