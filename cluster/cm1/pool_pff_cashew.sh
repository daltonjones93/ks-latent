#!/bin/bash
# 10e: particle flow filter in the pooled DA twin (scripts/cm1/da_pool.py --systems single_pff / pool_pff), same G0 setup and
# scores as pool_da_cashew.sh. single_pff: model 0 x 32, domain-localized latent PFF (ks_latent/cm1/pff_site.py). pool_pff: one PFF per
# model, each on all 32 members (its own 8 latents + the other 24 decoded members encoded by it); each model keeps its own 8.
# Timing (Cashew, 16 cores): 54 s per single analysis, 212 s per pool analysis (36 cycles: ~0.6 h / ~2.2 h per storm and setting).
# Stage 1 (this script, "val"): RTPS 0.3 / 0.6 / 0.9 on the 6 validation storms; pool_pff one job per storm and RTPS (18), single_pff
# 2 jobs x 3 storms x 3 RTPS. Stage 2 ("test_pool RTPS FIRST LAST", "test_single RTPS"): the chosen RTPS on the 22 test storms.
#   ssh cashew 'cd ~/latent_DA && bash cluster/cm1/pool_pff_cashew.sh val'
set -euo pipefail
cd "$HOME/latent_DA"
R=$HOME/cm1_runs
M="$R/F6_conv416_mixer512_gelu_pred_all_stagec $R/F6s2_stagec $R/F6s3_stagec $R/F6s5_stagec"
ENV=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16; cd \$HOME/latent_DA"
run() { sbatch --parsable -p coe_cpu --cpus-per-task=16 --mem=48G --time=$2 --job-name=$1 --output=logs/$1_%j.out --wrap="$ENV; $3"; }
O=\$HOME/cm1_da
if [ "$1" = val ]; then
  for rt in 0.3 0.6 0.9; do for j in $(seq 0 5); do
    run ppff_v${rt}_$j 02:50:00 "python scripts/cm1/da_pool.py --runs $M --split val --systems pool_pff --rtps $rt --job $j --n-jobs 6 --out $O/pff_val/pool_$rt"
  done; done
  for j in 0 1; do
    run spff_v$j 07:00:00 "python scripts/cm1/da_pool.py --runs $M --split val --systems single_pff --rtps 0.3 0.6 0.9 --job $j --n-jobs 2 --out $O/pff_val/single"
  done
elif [ "$1" = test_pool ]; then          # test_pool RTPS FIRST LAST (job indices of 22; batches under the 20-job cap)
  for j in $(seq $3 $4); do               # validation jobs took up to 2 h 49 -> 3 h 40
    run ppff_t$j 03:40:00 "python scripts/cm1/da_pool.py --runs $M --split test --systems pool_pff --rtps $2 --job $j --n-jobs 22 --out $O/pff_test/pool"
  done
elif [ "$1" = test_single ]; then        # test_single RTPS
  for j in $(seq 0 3); do
    run spff_t$j 05:00:00 "python scripts/cm1/da_pool.py --runs $M --split test --systems single_pff --rtps $2 --job $j --n-jobs 4 --out $O/pff_test/single"
  done
fi
