#!/bin/bash
# 10c: cycled DA twin with the 4-model pool (scripts/cm1/da_pool.py), G0 setup (theta' + reflectivity + radial velocity every 2 km,
# 5-min cycles for 3 h, held boundaries, loc 5 km, 32 members). RTPS grid 0.3 / 0.6 / 0.9 for every system; the value is chosen
# per system on the 6 validation storms (lowest second-half prior nMSE) and reported on the 22 test storms (all values are run on
# test in parallel only to save time; the choice uses validation alone). Re-encoded latents are clamped to each model's training range
# (+10% margin; one seed's encoder sent analysis columns to |z| ~ 300 and the pooled run diverged without it). 6 validation jobs + 14 test jobs (Cashew cap: 20).
#   ssh cashew 'cd ~/latent_DA && bash cluster/cm1/pool_da_cashew.sh'
set -euo pipefail
cd "$HOME/latent_DA"
R=$HOME/cm1_runs
M="$R/F6_conv416_mixer512_gelu_pred_all_stagec $R/F6s2_stagec $R/F6s3_stagec $R/F6s5_stagec"
ENV=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16; cd \$HOME/latent_DA"
run() { sbatch --parsable -p coe_cpu --cpus-per-task=16 --mem=48G --time=$2 --job-name=$1 --output=logs/$1_%j.out --wrap="$ENV; $3"; }
J=""
for j in $(seq 0 5); do
  J="$J $(run pda_val$j 02:00:00 "python scripts/cm1/da_pool.py --runs $M --split val --rtps 0.3 0.6 0.9 --seeds 0 --job $j --n-jobs 6 --out \$HOME/cm1_da/pool_da_val_clamp")"
done
for j in $(seq 0 13); do
  J="$J $(run pda_test$j 03:30:00 "python scripts/cm1/da_pool.py --runs $M --split test --rtps 0.3 0.6 0.9 --seeds 0 --job $j --n-jobs 14 --out \$HOME/cm1_da/pool_da_test_clamp")"
done
echo "jobs$J"
