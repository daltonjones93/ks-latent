#!/bin/bash
# 10b figures: vote of the 4-model pool (F6, seeds 2, 3, 5) against ONE model with 8 and 32 members, three unseen storms, +15/+30/+60 min.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/pool_figs.sh'
set -euo pipefail
cd "$HOME/latent_DA"
R=$HOME/cm1_runs; RUNS="$R/F6_conv416_mixer512_gelu_pred_all_stagec,$R/F6s2_stagec,$R/F6s3_stagec,$R/F6s5_stagec"
for s in mem029 ens_A2_mem001 ens_B_mem016; do
sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=64G --time=01:00:00 --job-name=poolfig_$s --output=logs/poolfig_${s}_%j.out \
  --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA; \
python scripts/cm1/make_pool_figs.py --runs $RUNS --storm $s --cutoffs docs/fpe_report/data/vote_cutoffs.json --out \$HOME/cm1_da/figs_pool"
done
