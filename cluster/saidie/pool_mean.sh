#!/bin/bash
# 10b: the AVERAGE of the independently trained models (no thresholding): statistics on the 22 test storms x 3 starts (mean_eval.py: pool
# mean 4 x 8 vs one model's mean of 8 / 32, one member, the deterministic twin F7) and one animation per variable on mem029.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/pool_mean.sh'
set -euo pipefail
cd "$HOME/latent_DA"
R=$HOME/cm1_runs; F6=$R/F6_conv416_mixer512_gelu_pred_all_stagec
M="$F6 $R/F6s2_stagec $R/F6s3_stagec $R/F6s5_stagec"; RUNS=$(echo $M | tr ' ' ',')
ENV=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA"
cpu() { sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=$3 --time=$2 --job-name=$1 --output=logs/$1_%j.out --wrap="$ENV; $4"; }
e=$(cpu mean_eval 03:00:00 120G "python scripts/cm1/mean_eval.py --models $M --members 8 --det $R/F7_conv416_mixer512_det_stagec --split test --out \$HOME/cm1_da/mean_eval_test.json")
A=""
for v in u w th qv prs qliq qice; do
  A="$A $(cpu meananim_$v 01:30:00 64G "python scripts/cm1/make_mean_anim.py --run $RUNS --storm mem029 --vars $v --out \$HOME/cm1_da/anim_mean")"
done
echo "stats $e; animations$A"
