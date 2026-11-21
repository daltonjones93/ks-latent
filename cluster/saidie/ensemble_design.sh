#!/bin/bash
# 10b: multi-model ensemble design (scripts/cm1/ensemble_design.py) after the six F6 seeds exist. Grid: noise temperature 0.7 / 1.0 / 1.3 x
# initial-latent noise 0 / 0.1; pools of 1-6 models x 8 members and the single-model control with the same member counts; K-of-N,
# neighbourhood (0 / 1 / 2 / 4 km) and probability-matched votes for all 12 features. Validation storms (choose) and test storms (report).
#   AFTER=<jobids colon-separated> bash -l cluster/saidie/ensemble_design.sh
set -euo pipefail
cd "$HOME/latent_DA"
R=$HOME/cm1_runs
M="$R/F6_conv416_mixer512_gelu_pred_all_stagec $R/F6s1_stagec $R/F6s2_stagec $R/F6s3_stagec $R/F6s4_stagec $R/F6s5_stagec"
for T in 0.7 1.0 1.3; do for I in 0.0 0.1; do for SP in val test; do
  [ $SP = val ] && TL=01:00:00 || TL=03:15:00
  sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=120G --time=$TL --job-name=design_${SP} \
    --output=logs/design_${SP}_T${T}_I${I}_%j.out ${AFTER:+--dependency=afterok:$AFTER} \
    --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA; \
python scripts/cm1/ensemble_design.py --models $M --members 8 --temp $T --ic-noise $I --split $SP --out \$HOME/cm1_da/design/${SP}_T${T}_I${I}.json"
done; done; done
