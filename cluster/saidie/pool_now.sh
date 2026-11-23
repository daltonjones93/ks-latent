#!/bin/bash
# 10b: the consensus-of-independently-trained-models measurement with the seeds that exist (F6 = seed 0, seeds 2, 3, 5), without waiting for seed 1.
# 66 cases (22 test storms x 3 starts), 24 members everywhere: each single seed (24 of its own members), pool3 = F6 + s2 + s3 (8 each),
# pool4 = F6 + s2 + s3 + s5 (6 each); lags to 90 min, true boundaries. Three parallel CPU jobs (a single model with 32 members took 46 min, so
# six runs in one job would need ~4 h): out dirs pool_now_{a,b,c}; merged by reading the three ts_eval.json files.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/pool_now.sh'
set -euo pipefail
cd "$HOME/latent_DA"
R=$HOME/cm1_runs; F6=$R/F6_conv416_mixer512_gelu_pred_all_stagec
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
run() { sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=120G --time=02:30:00 --job-name=pool_now_$1 --output=logs/pool_now_$1_%j.out \
  --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA; \
python scripts/cm1/twoscale_eval.py --runs $2 --source $SRC --n-test 22 --threads 16 --ref-norm-run $R/um_local_stagec --lags 0,5,15,30,60,90 --samples 24 --out \$HOME/cm1_da/pool_now_$1"; }
run a "$F6 $R/F6s2_stagec"
run b "$R/F6s3_stagec $R/F6s5_stagec"
run c "pool3=$F6+$R/F6s2_stagec+$R/F6s3_stagec pool4=$F6+$R/F6s2_stagec+$R/F6s3_stagec+$R/F6s5_stagec"
