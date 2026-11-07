#!/bin/bash -l
# Test-storm DA study at the model noise chosen on the validation storms (pick_tuned.py), then a paired
# comparison with the headline (full_v1_stagec, noise 0.1, ~/cm1_da/noise_det_0.1).
#   sbatch --export=ALL,RUN=...,LABEL=...,NOISES="0.05 0.1",DATA=... cluster/saidie/da_tuned.slurm
#SBATCH --job-name=cm1datuned
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/cm1datuned_%j.out
set -euo pipefail
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES not set: refusing to run on an unassigned GPU"; exit 1; }
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8
cd "$REPO"
dirs=""; for n in ${NOISES:?}; do dirs="$dirs $HOME/cm1_da/tune_${LABEL:?}_noise_$n"; done
noise=$(python scripts/cm1/pick_tuned.py --record "$HOME/cm1_da/tune_${LABEL}_choice.json" $dirs | tee /dev/stderr | tail -1)
echo "chosen on validation: model noise $noise"
OUT=$HOME/cm1_da/da_${LABEL}
python scripts/cm1/da_study.py --run "$HOME/cm1_runs/${RUN:?}" --source "${DATA:?}" --out "$OUT" --device cuda \
    --include-test --modes latent --n-members 32 --random-members --seeds 0 1 2 --obs-set th+refl+vr \
    --boundary persistence --model-noise "$noise"
python scripts/cm1/compare_studies.py "$HOME/cm1_da/noise_det_0.1" "$OUT"
