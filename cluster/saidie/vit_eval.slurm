#!/bin/bash -l
# 8a evaluation (CPU): reconstruction (lag 0) and forecast scores of the ViT pilot against the conv-autoencoder controls.
#SBATCH --job-name=v_eval
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:15:00
#SBATCH --output=logs/v_eval_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=""
cd "$REPO"
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
R=$HOME/cm1_runs
L=""; for r in ${RUNS:?} g_local48_stagec full_v1_stagec_crps; do [ -f "$R/$r/prop.pt" ] && L="$L $R/$r" || echo "missing $r"; done
python scripts/cm1/twoscale_eval.py --runs $L --source "$SRC" --out "$HOME/cm1_da/${OUT:-v_eval}" --lags 0,5,15,30,60 --threads 16 ${EXTRA_EVAL:-}
