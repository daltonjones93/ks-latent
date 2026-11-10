#!/bin/bash
#   sbatch -p coe_cpu --cpus-per-task=8 --mem=48G -t 02:00:00 -o logs/prop_sens_%j.out --export=ALL,RUN=ens_A_cpu_k45 kit/prop_sensitivity.sh
set -e
. ~/venvs/cm1/bin/activate
export PYTHONPATH=$HOME/latent_DA OMP_NUM_THREADS=8
cd ~/latent_DA
python scripts/cm1/prop_sensitivity.py --run ~/cm1_runs/${RUN:?} --base ~/cm1_cache_P --branches ~/cm1_cache_Q \
    --manifest ~/cm1_bundle/ens_Q/manifest.json --out ~/cm1_da/sens_${RUN} --threads 8
