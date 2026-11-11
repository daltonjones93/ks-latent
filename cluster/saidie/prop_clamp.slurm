#!/bin/bash -l
# Channel clamping (pre-registration 5o). CPU only. Ordered model on its test split; unordered control on ens_A2 members.
#   cd ~/latent_DA && sbatch cluster/saidie/prop_clamp.slurm
#SBATCH --job-name=clamp
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/clamp_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
cd "$REPO"
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
python scripts/cm1/prop_clamp.py --run "$HOME/cm1_runs/fpc_nested_stagec" --source "$SRC" --out "$HOME/cm1_da/clamp_nested" --threads 8
python scripts/cm1/prop_clamp.py --run "$HOME/cm1_runs/d34_c32_kuniform" --source "$SRC" --names-prefix ens_A2 --out "$HOME/cm1_da/clamp_unordered" --threads 8
