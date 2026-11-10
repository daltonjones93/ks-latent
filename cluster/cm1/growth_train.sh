#!/bin/bash
# Growth-matching fine-tune (5n). FOLD 1: train A029,B001 / test A030,B002; FOLD 2 the swap. RUN = base model run.
#   sbatch -p coe_cpu --cpus-per-task=8 --mem=32G -t 03:00:00 -o logs/growth_%j.out --export=ALL,RUN=d34_c32_kuniform,FOLD=1 kit/growth_train.sh
set -e
. ~/venvs/cm1/bin/activate
export PYTHONPATH=$HOME/latent_DA OMP_NUM_THREADS=8
cd ~/latent_DA
if [ "${FOLD:?}" = 1 ]; then TR=memA029,memB001; TE=memA030,memB002; else TR=memA030,memB002; TE=memA029,memB001; fi
python scripts/cm1/growth_train.py --run ~/cm1_runs/${RUN:?} --base ~/cm1_cache_P --branches ~/cm1_cache_Q \
    --manifest ~/cm1_bundle/ens_Q/manifest.json --train $TR --test $TE --out ~/cm1_runs/${RUN}_growth_f${FOLD} \
    --threads 8 ${EXTRA}
