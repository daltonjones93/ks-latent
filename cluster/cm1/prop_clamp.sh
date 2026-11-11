#!/bin/bash
# Channel clamping (5o). RUN = model run under ~/cm1_runs; SRC = cache dirs; PREFIX optional member-name prefix.
#   sbatch -p coe_cpu --cpus-per-task=8 --mem=32G -t 02:00:00 -o ~/logs/clamp_%j.out --export=ALL,RUN=fpc_nested_stagec,SRC=cache:~/cm1_cache kit/prop_clamp.sh
set -e
. ~/venvs/cm1/bin/activate
export PYTHONPATH=$HOME/latent_DA OMP_NUM_THREADS=8
cd ~/latent_DA
python scripts/cm1/prop_clamp.py --run ~/cm1_runs/${RUN:?} --source ${SRC:?} --out ~/cm1_da/clamp_${RUN} --threads 8 ${EXTRA}
