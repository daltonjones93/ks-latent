#!/bin/bash -l
# G1/G2/G3 latent DA study (many truths, baselines, forecasts from analyses) on one Saidie GPU.
#   cd ~/latent_DA && sbatch --export=ALL,RUN=d34_c32_kuniform,OUT=g1_v1,DATA=cache:$HOME/cm1_cache+$HOME/cm1_cache_A2,EXTRA="--truth-prefix ens_A2_ --n-truths 10 --include-test" cluster/saidie/da_study.slurm
#SBATCH --job-name=cm1study
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cm1study_%j.out
set -euo pipefail
# Saidie shows every GPU to every job; only CUDA_VISIBLE_DEVICES keeps torch on the GPU Slurm gave us.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES not set: refusing to run on an unassigned GPU"; exit 1; }
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8
cd "$REPO"
python -m pytest -q tests/unit/test_cm1*.py -p no:cacheprovider
python scripts/cm1/da_study.py --run "$HOME/cm1_runs/${RUN:?set RUN}" --source "${DATA:-cache:$HOME/cm1_cache}" \
    --out "$HOME/cm1_da/${OUT:?set OUT}" --device cuda ${EXTRA:-}
