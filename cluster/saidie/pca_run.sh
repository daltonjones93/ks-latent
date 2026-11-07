#!/bin/bash
# I-PCA (1): linear local-latent baseline. Fixed PCA-32 of each 2 km site's column block (fitted on the
# training storms) in place of the autoencoder; same split, same stencil propagator recipe (Stage B, then
# Stage C on the propagator only: PCA has no trainable parameters), then DA under the G0 protocol
# (da_eval_run.sh: validation-tuned noise, 22 test storms, paired against the headline). One GPU, chained.
#   ssh saidie 'cd ~/latent_DA && AFTER=<job> bash -l cluster/saidie/pca_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --ae-kind pca"
R=$HOME/cm1_runs
dep=${AFTER:+--dependency=afterany:$AFTER}
j1=$(sbatch --parsable ${dep:-} --mem=200G --time=04:00:00 --export=ALL,DATA="$DATA",RUN=pca32,EXTRA="$BASE" cluster/saidie/train_cm1.slurm)
j2=$(sbatch --parsable --dependency=afterok:$j1 --mem=200G --time=03:00:00 --export=ALL,DATA="$DATA",RUN=pca32_stagec,EXTRA="$BASE --init-from $R/pca32 --reuse-ae $R/pca32 --stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2" cluster/saidie/train_cm1.slurm)
echo "pca32 $j1, pca32_stagec $j2"
AFTER=$j2 bash cluster/saidie/da_eval_run.sh pca32_stagec pca32
