#!/bin/bash
# (Run once on Cashew.) Python environment for CM1 training on Cashew's GPU node:
# anaconda's Python 3.11 + a venv with torch (CUDA wheels), numpy, matplotlib, pytest.
# Usage: bash ~/latent_DA/cluster/cashew_gpu/setup_env.sh
set -euo pipefail
module load anaconda/24.1.2
ENV=${ENV:-$HOME/venvs/cm1}
[ -x "$ENV/bin/python" ] || python -m venv "$ENV"
. "$ENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet torch numpy matplotlib pytest
python -c "import torch, numpy; print('torch', torch.__version__, 'CUDA build', torch.version.cuda, 'numpy', numpy.__version__)"
echo "environment ready: $ENV"
