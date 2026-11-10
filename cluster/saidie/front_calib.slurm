#!/bin/bash -l
#   cd ~/latent_DA && sbatch cluster/saidie/front_calib.slurm
#SBATCH --job-name=frontcal
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/frontcal_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
cd "$REPO"
python scripts/cm1/front_detect.py --source "cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B" --out "$HOME/cm1_da/front_calib" --n 40
