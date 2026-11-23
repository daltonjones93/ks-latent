#!/bin/bash
# 10b: the consensus-of-independently-trained-models measurement with the seeds that exist (F6 = seed 0, seeds 2, 3, 5), without waiting for seed 1.
# 66 cases (22 test storms x 3 starts), 24 members everywhere: each single seed (24 of its own members), pool3 = F6 + s2 + s3 (8 each),
# pool4 = F6 + s2 + s3 + s5 (6 each); lags to 90 min, true boundaries.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/pool_now.sh'
set -euo pipefail
cd "$HOME/latent_DA"
R=$HOME/cm1_runs; F6=$R/F6_conv416_mixer512_gelu_pred_all_stagec
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=120G --time=03:00:00 --job-name=pool_now --output=logs/pool_now_%j.out \
  --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA; \
python scripts/cm1/twoscale_eval.py --runs $F6 $R/F6s2_stagec $R/F6s3_stagec $R/F6s5_stagec pool3=$F6+$R/F6s2_stagec+$R/F6s3_stagec pool4=$F6+$R/F6s2_stagec+$R/F6s3_stagec+$R/F6s5_stagec \
--source $SRC --n-test 22 --threads 16 --ref-norm-run $R/um_local_stagec --lags 0,5,15,30,60,90 --samples 24 --out \$HOME/cm1_da/pool_now"
