#!/bin/bash
# 10b: the ensemble design run on the four finished models (F6, seeds 2, 3, 5), test storms, default noise (temperature 1, no initial-state noise),
# 8 members per model, pools of 1-4 models and the single-model control; the animation's vote rule is then applied by vote_pool_report.py.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/design_now.sh'
set -euo pipefail
cd "$HOME/latent_DA"
R=$HOME/cm1_runs; F6=$R/F6_conv416_mixer512_gelu_pred_all_stagec
sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=120G --time=03:30:00 --job-name=design_now --output=logs/design_now_%j.out \
  --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA; \
python scripts/cm1/ensemble_design.py --models $F6 $R/F6s2_stagec $R/F6s3_stagec $R/F6s5_stagec --members 8 --temp 1.0 --ic-noise 0.0 --split test --out \$HOME/cm1_da/design/now_test_T1.0_I0.0.json"
