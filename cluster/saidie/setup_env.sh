#!/bin/bash
# (Run once on Saidie.) A small Python environment for CM1 training: torch, numpy, matplotlib, pytest,
# in a venv on the cluster's python312 module (Saidie's system Python has no ensurepip, so it cannot make venvs). TORCH_INDEX picks the torch build for the node's driver, e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash ~/latent_DA/cluster/saidie/setup_env.sh
# (default: PyPI's torch). Re-running reuses the venv.
set -euo pipefail
if type module >/dev/null 2>&1 || [ -f /etc/profile.d/modules.sh ]; then
  set +u   # the module init scripts read unset variables
  type module >/dev/null 2>&1 || . /etc/profile.d/modules.sh
  module load python312 2>/dev/null || true
  set -u
fi
python3 -c "import ensurepip" || { echo "this python3 ($(which python3)) cannot make venvs (no ensurepip)"; exit 1; }
ENV=${ENV:-$HOME/venvs/cm1}
[ -x "$ENV/bin/python" ] || python3 -m venv "$ENV"
. "$ENV/bin/activate"
pip install --quiet --upgrade pip
if [ -n "${TORCH_INDEX:-}" ]; then
  pip install --quiet torch --index-url "$TORCH_INDEX"
else
  pip install --quiet torch
fi
pip install --quiet numpy matplotlib pytest
python -c "import torch, numpy; print('torch', torch.__version__, 'CUDA build', torch.version.cuda, 'numpy', numpy.__version__)"
echo "environment ready: $ENV"
