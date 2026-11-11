#!/bin/bash -l
# Evaluate the pruned and fine-tuned runs (5s) on the 5r cases. CPU only.
#SBATCH --job-name=pfxeval
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pfxeval_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
cd "$REPO"
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
R=$HOME/cm1_runs
python scripts/cm1/prefix_eval.py --source "$SRC" --out "$HOME/cm1_da/prefix_dft" --threads 8 --runs \
  $R/fpc_nested_ctl8k_full $R/fpc_nested_ctl8k_prop \
  $R/fpc_nested_p4_dft:distilled $R/fpc_nested_p8_dft:distilled $R/fpc_nested_p12_dft:distilled $R/fpc_nested_p16_dft:distilled \
  $R/fpc_nested_p4_dft $R/fpc_nested_p8_dft $R/fpc_nested_p12_dft $R/fpc_nested_p16_dft
