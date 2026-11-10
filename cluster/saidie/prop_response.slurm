#!/bin/bash -l
# Propagator response function and linearisation (pre-registration 5j). CPU only.
#   cd ~/latent_DA && sbatch cluster/saidie/prop_response.slurm
#SBATCH --job-name=propresp
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/propresp_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
cd "$REPO"
python scripts/cm1/prop_response.py --run "$HOME/cm1_runs/fpc_nested_stagec" \
    --source "cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B" --out "$HOME/cm1_da/prop_resp" --threads 8
