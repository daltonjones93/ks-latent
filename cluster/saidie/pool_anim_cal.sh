#!/bin/bash
# 10b: per-feature/per-lead vote cutoffs calibrated for the 4-model pool (32 members) on the VALIDATION storms (6 storms x 3 starts,
# leads 5-90 min), then the pooled vote animations re-rendered with them (file tag _cal).
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/pool_anim_cal.sh'
set -euo pipefail
cd "$HOME/latent_DA"
R=$HOME/cm1_runs; F6=$R/F6_conv416_mixer512_gelu_pred_all_stagec
M="$F6 $R/F6s2_stagec $R/F6s3_stagec $R/F6s5_stagec"; RUNS=$(echo $M | tr ' ' ',')
ENV=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA"
CUT=\$HOME/cm1_da/design/vote_cutoffs_pool4.json
d=$(sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=120G --time=01:45:00 --job-name=cal_pool4 --output=logs/cal_pool4_%j.out \
  --wrap="$ENV; python scripts/cm1/ensemble_design.py --models $M --members 8 --temp 1.0 --ic-noise 0.0 --split val --no-single \
--lags 5,10,15,20,30,45,60,75,90 --out \$HOME/cm1_da/design/pool4_val_cal.json && \
python scripts/cm1/calibrate_pool_cutoffs.py --val \$HOME/cm1_da/design/pool4_val_cal.json --out $CUT")
A=""
for v in u w th qv prs qliq qice; do
A="$A $(sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=64G --time=01:30:00 --dependency=afterok:$d --job-name=poolanimc_$v --output=logs/poolanimc_${v}_%j.out \
  --wrap="$ENV; python scripts/cm1/make_vote_anim.py --run $RUNS --storm mem029 --vars $v --cutoffs $CUT --cutoffs8 docs/fpe_report/data/vote_cutoffs.json --tag _cal --out \$HOME/cm1_da/anim_pool")"
done
echo "calibration $d; animations$A"
