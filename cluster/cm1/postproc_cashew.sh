#!/bin/bash
# 10b: post-processing of the pooled forecast (align / local align / probability matching / scale blends / medoid) on the 22 test storms,
# 11 parallel Cashew CPU jobs of 2 storms each (scripts/cm1/postproc_eval.py); merge with scripts/cm1/postproc_report.py.
#   ssh cashew 'cd ~/latent_DA && bash cluster/cm1/postproc_cashew.sh'
set -euo pipefail
cd "$HOME/latent_DA"
R=$HOME/cm1_runs
M="$R/F6_conv416_mixer512_gelu_pred_all_stagec $R/F6s2_stagec $R/F6s3_stagec $R/F6s5_stagec"
for j in $(seq 0 10); do
  sbatch --parsable -p coe_cpu --cpus-per-task=16 --mem=48G --time=01:30:00 --job-name=pp_$j --output=logs/pp_${j}_%j.out \
    --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16; cd \$HOME/latent_DA; \
python scripts/cm1/postproc_eval.py --models $M --members 8 --split test --job $j --n-jobs 11 --out \$HOME/cm1_da/postproc"
done
