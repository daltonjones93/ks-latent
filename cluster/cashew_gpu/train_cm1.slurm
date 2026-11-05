#!/bin/bash
# CM1 column local field training on Cashew's GPU node (coe_gpu, 1 GPU), data already on Cashew.
#   cd ~/latent_DA && sbatch --export=ALL,RUN=ens_A_v1 cluster/cashew_gpu/train_cm1.slurm
# Data stay in host memory (--data-device cpu): 31 members x 361 frames is ~15 GB in float16,
# more than some GPUs hold; only batches go to the GPU.
#SBATCH --job-name=cm1train
#SBATCH --partition=coe_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cm1train_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to torch'; print(torch.cuda.get_device_name(0))"
cd "$REPO"
python -m pytest -q tests/unit/test_cm1.py -p no:cacheprovider     # refuse to train on a broken install
python scripts/cm1/train_cm1.py --profile full --source "grads:${DATA:-$HOME/cm1_bundle/ens_A}" --device cuda \
    --data-device cpu --threads 8 --cache "$HOME/cm1_cache" --out "$HOME/cm1_runs/${RUN:-run_$SLURM_JOB_ID}" ${EXTRA:-}
