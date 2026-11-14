#!/bin/bash -l
# CM1 column local field training on one Saidie H200.
#   cd ~/latent_DA && sbatch --export=ALL,DATA="cache:$HOME/cm1_cache",RUN=ens_A_v1 cluster/saidie/train_cm1.slurm
# DATA is any train_cm1.py --source: cache:DIR[:PATTERN] (the float16 member files copied from Cashew),
# grads:DIR, or synthetic. EXTRA passes further options, e.g. EXTRA="--k-max 45".
# One GPU per job; the node (8 x H200) is shared.
#SBATCH --job-name=cm1train
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00   # default; sbatch --time overrides (the 7a jobs took 19-63 min; a 12 h request slows backfill scheduling)
#SBATCH --output=logs/cm1train_%j.out
set -euo pipefail
# Saidie shows every GPU to every job; only CUDA_VISIBLE_DEVICES keeps torch on the GPU Slurm gave us.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES not set: refusing to run on an unassigned GPU"; exit 1; }
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=16
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to torch'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
cd "$REPO"
python -m pytest -q tests/unit/test_cm1.py -p no:cacheprovider     # refuse to train on a broken install
python scripts/cm1/train_cm1.py --profile full --source "${DATA:?set DATA, e.g. cache:\$HOME/cm1_cache}" --device cuda \
    --threads 16 --cache "$HOME/cm1_cache_build" --out "$HOME/cm1_runs/${RUN:-run_$SLURM_JOB_ID}" ${EXTRA:-}
