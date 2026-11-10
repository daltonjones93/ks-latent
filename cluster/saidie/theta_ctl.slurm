#!/bin/bash -l
# FPe control: the channel-0 library fitted directly to site-mean surface theta' (CPU only, no autoencoder in the fit).
#   cd ~/latent_DA && sbatch cluster/saidie/theta_ctl.slurm
#SBATCH --job-name=thctl
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/thctl_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
cd "$REPO"
python scripts/cm1/law_theta_control.py --run "$HOME/cm1_runs/fpc_nested_stagec" \
    --source "cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B" --out "$HOME/cm1_da/fpe_thctl"
