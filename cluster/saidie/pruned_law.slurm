#!/bin/bash -l
# 5u: explicit law of the pruned models vs the unpruned model restricted to the same channels. CPU only.
#SBATCH --job-name=prunedlaw
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prunedlaw_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
cd "$REPO"
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
R=$HOME/cm1_runs
python scripts/cm1/pruned_law.py --base $R/fpc_nested_stagec --source "$SRC" --m 8 --out "$HOME/cm1_da/pruned_law_m8" --threads 8 \
  --runs $R/fpc_nested_p8_prop $R/fpc_nested_p8_full $R/fpc_nested_p8_dft
