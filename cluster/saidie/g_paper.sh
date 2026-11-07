#!/bin/bash
# Paper III: the remaining tier-1 DA runs (docs/cm1_plan.md, G "claims status"). ONE Saidie GPU, sequential,
# after AFTER (default: S55, 9014). Model: full_v1_stagec, the G0 headline setup (22 test storms x 3 seeds,
# theta' + reflectivity + radial velocity, 32 random training members, loc 5 km, RTPS 0.9, boundaries held at t0).
#   1. g3_headline     the headline (latent, noise 0.1) rerun with forecasts from the analyses at 90 / 120 / 150 /
#                      180 min (G3: DA-initialized vs no-DA forecasts vs persistence of the analysis, reflectivity
#                      FSS) and the per-cycle ranks for claim 2's rank-histogram test (rank_test.py)
#   2. G2b baselines, same LETKF and observations, other forecast models; noise chosen on the validation storms
#      (G0 addendum 3): persistence ensemble 0.1 / 0.2 / 0.3, local analog (one drawn neighbour of 10) 0 / 0.05 /
#      0.1 -- tuning runs first, then every candidate on the test storms (the report uses the validation choice)
#   ssh saidie 'cd ~/latent_DA && AFTER=9014 bash -l cluster/saidie/g_paper.sh'
set -euo pipefail
cd "$HOME/latent_DA"
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
RUN=full_v1_stagec
SET="--modes MODE --n-members 32 --random-members --seeds 0 1 2 --obs-set th+refl+vr --boundary persistence"
prev=${AFTER-9014}
sub() {   # out, extra
  prev=$(sbatch --parsable ${prev:+--dependency=afterany:$prev} --export=ALL,RUN=$RUN,DATA="$DATA",OUT=$1,EXTRA="$2" cluster/saidie/da_study.slurm)
  echo "$1: $prev"
}
sub g3_headline "${SET/MODE/latent} --include-test --model-noise 0.1 --forecast-from 5 11 17 23"
for n in 0.1 0.2 0.3; do sub g2b_persist_val_$n "${SET/MODE/persist} --tuning --model-noise $n --forecast-from"; done
for n in 0 0.05 0.1; do sub g2b_analog_val_$n "${SET/MODE/analog} --tuning --model-noise $n --forecast-from"; done
for n in 0.1 0.2 0.3; do sub g2b_persist_$n "${SET/MODE/persist} --include-test --model-noise $n --forecast-from"; done
for n in 0 0.05 0.1; do sub g2b_analog_$n "${SET/MODE/analog} --include-test --model-noise $n --forecast-from"; done
