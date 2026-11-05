#!/bin/bash -l
# CM1 unit tests and the smoke profile on a Saidie GPU, on synthetic data (no CM1 data needed).
#   cd ~/latent_DA && sbatch cluster/saidie/smoke.slurm
#SBATCH --job-name=cm1smoke
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/cm1smoke_%j.out
set -euo pipefail
# Saidie shows every GPU to every job; only CUDA_VISIBLE_DEVICES keeps torch on the GPU Slurm gave us.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES not set: refusing to run on an unassigned GPU"; exit 1; }
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=4
cd "$REPO"
python -m pytest -q tests/unit/test_cm1.py -p no:cacheprovider
python scripts/cm1/train_cm1.py --profile smoke --source synthetic --device cuda --out "$HOME/cm1_runs/smoke_gpu"
python - <<'PY'
import json, os
r = json.load(open(os.path.expanduser("~/cm1_runs/smoke_gpu/results.json")))
print("SMOKE OK: results.json written; device check:", json.load(open(os.path.expanduser("~/cm1_runs/smoke_gpu/ae.pt.meta.json")))["device"])
PY
