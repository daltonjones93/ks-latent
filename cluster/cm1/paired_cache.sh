#!/bin/bash
#   sbatch -p coe_cpu --cpus-per-task=4 --mem=32G -t 01:00:00 -o logs/paired_cache_%j.out kit/paired_cache.sh
set -e
. ~/venvs/cm1/bin/activate
export PYTHONPATH=$HOME/latent_DA OMP_NUM_THREADS=4
cd ~/latent_DA
python scripts/cm1/build_cache.py ~/cm1_bundle/ens_Q ~/cm1_cache_Q
python - <<'PY'
from pathlib import Path
from ks_latent.cm1.data import load_member
f = sorted(Path.home().joinpath("cm1_cache_Q").glob("*ctrl.npz"))[0]
m = load_member(f); print(f.name, m.X.shape, "t:", m.t[:3], "...", m.t[-1])
PY
