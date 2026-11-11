#!/bin/bash -l
# Channel-0 characterisation (pre-registration 5q). CPU only.
#   cd ~/latent_DA && sbatch cluster/saidie/channel0_char.slurm
#SBATCH --job-name=ch0
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ch0_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
cd "$REPO"
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
python scripts/cm1/channel0_char.py --run "$HOME/cm1_runs/fpc_nested_stagec" --source "$SRC" --out "$HOME/cm1_da/ch0_nested" --threads 8
python scripts/cm1/channel0_char.py --run "$HOME/cm1_runs/d34_c32_kuniform" --source "$SRC" --names-prefix ens_A2 --out "$HOME/cm1_da/ch0_unordered" --threads 8
