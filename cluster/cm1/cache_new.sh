#!/bin/bash
# 8a: float16 caches of the new ensembles (ens_C, ens_D) on Cashew: ONE CPU job per ensemble (Cashew caps a partition at
# 20 submitted+pending jobs), running 8 build_cache shards side by side. Waits for the ensemble's CM1 jobs (afterany:
# build_cache skips unfinished members; rerun this script after resubmitting failed members).
#   ssh cashew 'cd ~/latent_DA && AFTER_C=j1:j2:... AFTER_D=j3:... bash cluster/cm1/cache_new.sh'
set -euo pipefail
cd "$HOME/latent_DA"; mkdir -p logs
for E in ens_C ens_D; do
  v=AFTER_${E#ens_}; dep=${!v:+--dependency=afterany:${!v}}
  out=$HOME/cm1_cache_${E#ens_}
  sbatch --parsable ${dep:-} -p coe_cpu --cpus-per-task=8 --mem=96G -t 12:00:00 -J cache_$E -o logs/cache_${E}_%j.out \
    --wrap ". ~/venvs/cm1/bin/activate; export PYTHONPATH=$HOME/latent_DA OMP_NUM_THREADS=1; seq 0 7 | xargs -P 8 -I{} python scripts/cm1/build_cache.py $HOME/cm1_bundle/$E $out --shard {} 8"
done
