#!/bin/bash -l
# Re-evaluate trained runs with the current metrics (train_cm1.py --eval-only), one after another on
# one GPU. Each run's own config.json fixes the architecture.
#   cd ~/latent_DA && sbatch --export=ALL,RUNS="ens_A_gpu_k45 d4_c32",DATA=cache:$HOME/cm1_cache cluster/saidie/rescore.slurm
#SBATCH --job-name=cm1rescore
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cm1rescore_%j.out
set -euo pipefail
# Saidie shows every GPU to every job; only CUDA_VISIBLE_DEVICES keeps torch on the GPU Slurm gave us.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES not set: refusing to run on an unassigned GPU"; exit 1; }
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8
cd "$REPO"
python -m pytest -q tests/unit/test_cm1.py tests/unit/test_cm1_verify.py -p no:cacheprovider
for r in ${RUNS:?set RUNS}; do
  echo "=== $r"
  python scripts/cm1/train_cm1.py --profile full --source "${DATA:?set DATA}" --device cuda --threads 8 \
      --out "$HOME/cm1_runs/$r" --eval-only
done
