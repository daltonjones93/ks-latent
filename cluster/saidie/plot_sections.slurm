#!/bin/bash -l
# x-z section figures for the best CM1 model (scripts/cm1/plot_sections.py); CPU only, a few minutes.
#   cd ~/latent_DA && sbatch cluster/saidie/plot_sections.slurm
#SBATCH --job-name=cm1fig
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cm1fig_%j.out
set -euo pipefail
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$HOME/latent_DA OMP_NUM_THREADS=8
cd "$HOME/latent_DA"
B=$HOME/cm1_runs/${RUN:-d34_c32_kuniform}
mkdir -p "$HOME/cm1_figs"
for v in "qliq qice" "th w"; do
  python scripts/cm1/plot_sections.py --run "$B" --average-props "${B}_prop_s1" "${B}_prop_s2" "${B}_prop_s3" \
      --source "cache:$HOME/cm1_cache" --member "${MEMBER:-mem029}" --vars $v \
      --out "$HOME/cm1_figs/sections_${v// /_}.png" --device cpu
done
