#!/bin/bash
#   sbatch -p coe_cpu --cpus-per-task=4 --mem=32G -t 01:00:00 -o logs/paired_cache_%j.out kit/paired_cache.sh
set -e
. ~/venvs/cm1/bin/activate
export PYTHONPATH=$HOME/latent_DA OMP_NUM_THREADS=4
cd ~/latent_DA
python scripts/cm1/build_branch_cache.py --base ~/cm1_bundle/ens_P --branches ~/cm1_bundle/ens_Q --out ~/cm1_cache_Q
python - <<'PY'
from pathlib import Path
from ks_latent.cm1.data import load_member
import numpy as np
b = load_member(sorted(Path.home().joinpath("cm1_cache_P").glob("*memA029.npz"))[0])
c = load_member(sorted(Path.home().joinpath("cm1_cache_Q").glob("*memA029_ctrl.npz"))[0])
print("base", b.X.shape, "t", b.t[-2:], "ctrl", c.X.shape, "t", c.t[:2])
# continuity: ctrl frame 1 (7260 s) vs base frame at 7260 s are different runs only by restart; compare the field means per variable
i = int(np.argmin(abs(b.t - 7260)))
print("max |ctrl(7260) - base(7260)| per var:", np.abs(c.X[0].astype(float) - b.X[i].astype(float)).max(axis=(0, 2)))
PY
