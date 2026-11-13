#!/bin/bash -l
# 7a evaluation (CPU): forecast scores with true and persistence boundaries, the global-off ablation, and the reach diagnostic.
#SBATCH --job-name=g_eval
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/g_eval_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=""
cd "$REPO"
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
R=$HOME/cm1_runs
G="$R/g_local48_stagec $R/g_local128_stagec $R/g_local128_s1_stagec $R/g_mean8_stagec $R/g_attn4_stagec $R/g_attn8_stagec $R/g_attn8_s1_stagec $R/g_attn32_stagec"
RUNS=""; for r in $G; do [ -f "$r/prop.pt" ] && RUNS="$RUNS $r" || echo "missing $r"; done
python scripts/cm1/twoscale_eval.py --runs $RUNS $R/full_v1_stagec_crps $R/ts_v1 --source "$SRC" --out "$HOME/cm1_da/g_eval" --global-off --threads 16
python scripts/cm1/twoscale_eval.py --runs $RUNS --source "$SRC" --out "$HOME/cm1_da/g_eval_pers" --boundary persistence --global-off --threads 16
python scripts/cm1/global_diag.py --runs $RUNS --source "$SRC" --out "$HOME/cm1_da/g_diag" --threads 16
