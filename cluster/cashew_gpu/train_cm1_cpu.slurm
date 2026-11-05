#!/bin/bash
# CM1 column local field training on a Cashew CPU node (coe_cpu): no GPU needed; the model is small
# (~3 M parameters). Same run as train_cm1.slurm, on 32 cores.
#   cd ~/latent_DA && sbatch --export=ALL,RUN=ens_A_cpu_v1 cluster/cashew_gpu/train_cm1_cpu.slurm
#SBATCH --job-name=cm1train_cpu
#SBATCH --partition=coe_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cm1train_cpu_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
cd "$REPO"
python -m pytest -q tests/unit/test_cm1.py -p no:cacheprovider     # refuse to train on a broken install
python scripts/cm1/train_cm1.py --profile full --source "grads:${DATA:-$HOME/cm1_bundle/ens_A}" --device cpu \
    --threads 32 --cache "$HOME/cm1_cache" --out "$HOME/cm1_runs/${RUN:-cpu_run_$SLURM_JOB_ID}" ${EXTRA:-}
