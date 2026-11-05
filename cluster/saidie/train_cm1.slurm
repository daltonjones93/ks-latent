#!/bin/bash
# CM1 column local field training on one Saidie GPU.
#   sbatch --export=ALL,DATA=/path/to/ens_A,RUN=ens_A_v1 ~/latent_DA/cluster/saidie/train_cm1.slurm
# FILL the account and partition (Saidie requires both; see `sacctmgr show assoc user=$USER` and `sinfo`).
#SBATCH --job-name=cm1train
#SBATCH --account=FILL_ACCOUNT
#SBATCH --partition=FILL_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cm1train_%j.out
set -euo pipefail
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$REPO"
python -m pytest -q tests/unit/test_cm1.py -p no:cacheprovider          # refuse to train on a broken install
python scripts/cm1/train_cm1.py --profile full --source "grads:${DATA:?set DATA=ensemble dir}" --device cuda \
    --threads 8 --cache "$HOME/cm1_cache" --out "$HOME/cm1_runs/${RUN:-run_$SLURM_JOB_ID}" ${EXTRA:-}
