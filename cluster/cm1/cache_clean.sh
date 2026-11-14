#!/bin/bash
# 8a: (KEEP=1: keep the raw output, a verification pass) cache + verify + delete raw output of the finished members of ENS (one 8-CPU job, 8 shards). Run after each wave of storms.
#   ssh cashew 'cd ~/latent_DA && bash cluster/cm1/cache_clean.sh ens_C [AFTER_JOBIDS]'
set -euo pipefail
E=$1; cd "$HOME/latent_DA"; mkdir -p logs
dep=${2:+--dependency=afterany:$2}
sbatch --parsable ${dep:-} -p coe_cpu --cpus-per-task=8 --mem=48G -t 03:00:00 -J cclean_$E -o logs/cclean_${E}_%j.out \
  --wrap ". ~/venvs/cm1/bin/activate; export PYTHONPATH=$HOME/latent_DA OMP_NUM_THREADS=1; seq 0 7 | xargs -P 8 -I{} python scripts/cm1/cache_clean.py $HOME/cm1_bundle/$E $HOME/cm1_cache_${E#ens_} --shard {} 8 ${KEEP:+--keep-raw} ${NOBUILD:+--no-build}"
