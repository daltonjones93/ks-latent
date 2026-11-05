#!/bin/bash
# D16 full run: all 121 storms (ens_A + ens_A2 + ens_B) with the screen's winning recipe (d34 recipe, then
# Stage C k <= 45). Two jobs chained on one GPU. Test: ens_A mem029-030, ens_A2 mem001-010 (the G1 DA
# truths) and the 10 ens_B screen test storms; val: ens_A mem028, ens_A2 mem011, the 4 ens_B val storms.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/full_run.sh [AFTER_JOBID]'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST"
R=$HOME/cm1_runs/full_v1
dep=${1:+--dependency=afterany:$1}
if [ -z "${STOCH_AFTER:-}" ]; then
j1=$(sbatch --parsable ${dep:-} --mem=200G --export=ALL,DATA="$DATA",RUN=full_v1,EXTRA="$BASE" cluster/saidie/train_cm1.slurm)
j2=$(sbatch --parsable --dependency=afterok:$j1 --mem=200G --export=ALL,DATA="$DATA",RUN=full_v1_stagec,EXTRA="$BASE --init-from $R --reuse-ae $R --stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2" cluster/saidie/train_cm1.slurm)
echo "full_v1 $j1, full_v1_stagec $j2"
fi
# Stochastic branch (added 2026-11-05; stochastic + Stage C wins on CRPS by ~30% on ens_A): a CRPS propagator
# on full_v1's autoencoder, then Stage C with decoded CRPS. Runs beside full_v1_stagec (2 GPUs at most).
#   ssh saidie 'cd ~/latent_DA && STOCH_AFTER=<full_v1 job> bash -l cluster/saidie/full_run.sh'
if [ -n "${STOCH_AFTER:-}" ]; then
  RS=$HOME/cm1_runs/full_v1_crps
  j3=$(sbatch --parsable --dependency=afterok:$STOCH_AFTER --mem=200G --export=ALL,DATA="$DATA",RUN=full_v1_crps,EXTRA="$BASE --reuse-ae $R --noise-channels 4 --crps-samples 4 --eval-samples 8" cluster/saidie/train_cm1.slurm)
  j4=$(sbatch --parsable --dependency=afterok:$j3 --mem=200G --export=ALL,DATA="$DATA",RUN=full_v1_stagec_crps,EXTRA="$BASE --init-from $RS --reuse-ae $RS --stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 6 --eval-samples 8" cluster/saidie/train_cm1.slurm)
  echo "full_v1_crps $j3, full_v1_stagec_crps $j4"
fi
