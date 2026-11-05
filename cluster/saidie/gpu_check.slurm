#!/bin/bash -l
# One-minute check of a Saidie GPU: driver, and (if the venv exists) whether torch sees CUDA.
#   cd ~/latent_DA && sbatch cluster/saidie/gpu_check.slurm
#SBATCH --job-name=cm1gpucheck
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu_check_%j.out
# Saidie shows every GPU to every job; only CUDA_VISIBLE_DEVICES keeps torch on the GPU Slurm gave us.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES not set: refusing to run on an unassigned GPU"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | grep -i "cuda version" || true
if [ -x "${ENV:-$HOME/venvs/cm1}/bin/python" ]; then
  "${ENV:-$HOME/venvs/cm1}/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
fi
