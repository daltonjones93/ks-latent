#!/bin/bash
# Paired runs, stage 2 (Cashew): branch directories from the stage-1 restarts (small job; nothing heavy on the login node).
#   sbatch -p coe_cpu --cpus-per-task=2 --mem=16G -t 00:30:00 -o logs/paired_make_%j.out kit/paired_make.sh
set -e
. ~/venvs/cm1/bin/activate
export PYTHONPATH=$HOME/latent_DA
cd ~/latent_DA
python scripts/cm1/make_paired.py --base ~/cm1_bundle/ens_P --cache ~/cm1_cache_P --out ~/cm1_bundle/ens_Q --detector min
ls ~/cm1_bundle/ens_Q/members | wc -l
