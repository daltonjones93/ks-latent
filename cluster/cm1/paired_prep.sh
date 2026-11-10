#!/bin/bash
# Paired runs, prep (Cashew): cache the stage-1 base members and calibrate the front detectors on the cached storms.
#   sbatch -p coe_cpu --cpus-per-task=4 --mem=48G -t 01:00:00 -o logs/paired_prep_%j.out kit/paired_prep.sh
set -e
. ~/venvs/cm1/bin/activate
export PYTHONPATH=$HOME/latent_DA OMP_NUM_THREADS=4
cd ~/latent_DA
python scripts/cm1/build_cache.py ~/cm1_bundle/ens_P ~/cm1_cache_P
python scripts/cm1/front_detect.py --source "cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B" --out ~/cm1_da/front_calib --n 40
