#!/bin/bash
# (Run once on Saidie.) A small Python environment for CM1 training: torch (CUDA wheels), numpy,
# matplotlib, pytest. Installing is light work, but if ETS prefers, run it inside a short job.
# Usage: bash ~/latent_DA/cluster/saidie/setup_env.sh
set -euo pipefail
ENV=${ENV:-$HOME/venvs/cm1}
python3 -m venv "$ENV"
. "$ENV/bin/activate"
pip install --upgrade pip
pip install torch numpy matplotlib pytest
python -c "import torch; print('torch', torch.__version__, 'CUDA build', torch.version.cuda)"
echo "environment ready: $ENV"
