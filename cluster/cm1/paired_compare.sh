#!/bin/bash
# Paired runs vs the propagator (Cashew CPU). RUN = model run directory under ~/cm1_runs.
#   sbatch -p coe_cpu --cpus-per-task=8 --mem=64G -t 04:00:00 -o logs/paired_compare_%j.out --export=ALL,RUN=ens_A_cpu_k45 kit/paired_compare.sh
set -e
. ~/venvs/cm1/bin/activate
export PYTHONPATH=$HOME/latent_DA OMP_NUM_THREADS=8
cd ~/latent_DA
python scripts/cm1/paired_compare.py --run ~/cm1_runs/${RUN:?} --base ~/cm1_cache_P --branches ~/cm1_cache_Q \
    --manifest ~/cm1_bundle/ens_Q/manifest.json --out ~/cm1_da/paired_${RUN} --threads 8
