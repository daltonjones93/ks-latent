#!/bin/bash
# 10b: vote animations for the pooled model (F6 + seeds 2, 3, 5; 4 x 8 = 32 members), one per variable, storm mem029 (as anim_vote/).
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/pool_anim.sh'
set -euo pipefail
cd "$HOME/latent_DA"
R=$HOME/cm1_runs; RUNS="$R/F6_conv416_mixer512_gelu_pred_all_stagec,$R/F6s2_stagec,$R/F6s3_stagec,$R/F6s5_stagec"
for v in u w th qv prs qliq qice; do
sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=64G --time=01:30:00 --job-name=poolanim_$v --output=logs/poolanim_${v}_%j.out \
  --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA; \
python scripts/cm1/make_vote_anim.py --run $RUNS --storm mem029 --vars $v --cutoffs docs/fpe_report/data/vote_cutoffs.json --out \$HOME/cm1_da/anim_pool"
done
