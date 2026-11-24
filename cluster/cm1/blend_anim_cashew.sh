#!/bin/bash
# 10b: scale-blend animations (scripts/cm1/make_blend_anim.py), 4-model pool x 8 members, mem029, one Cashew CPU job per variable.
#   ssh cashew 'cd ~/latent_DA && bash cluster/cm1/blend_anim_cashew.sh'
set -euo pipefail
cd "$HOME/latent_DA"
R=$HOME/cm1_runs
RUNS="$R/F6_conv416_mixer512_gelu_pred_all_stagec,$R/F6s2_stagec,$R/F6s3_stagec,$R/F6s5_stagec"
for v in u w th qv prs qliq qice; do
  sbatch --parsable -p coe_cpu --cpus-per-task=16 --mem=48G --time=01:30:00 --job-name=blend_$v --output=logs/blend_${v}_%j.out \
    --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16; cd \$HOME/latent_DA; \
python scripts/cm1/make_blend_anim.py --run $RUNS --storm mem029 --vars $v --out \$HOME/cm1_da/anim_blend"
done
