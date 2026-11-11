#!/bin/bash -l
# Prefix truncation (pre-registration 5r). CPU only.
#SBATCH --job-name=prefix
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prefix_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
cd "$REPO"
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
python scripts/cm1/prop_prefix.py --run "$HOME/cm1_runs/fpc_nested_stagec" --source "$SRC" --out "$HOME/cm1_da/prefix_nested" --threads 8
