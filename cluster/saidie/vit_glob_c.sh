#!/bin/bash
# 8a: v_glob's Stage C ran out of GPU memory (global attention on 200-site windows, batch 6, k <= 45); rerun with batch 2, then
# the pilot evaluation for it.   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/vit_glob_c.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
R=$HOME/cm1_runs
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32"
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 2 --c-freeze-ae"
rm -rf "$R/v_glob_stagec"
jc=$(sbatch --parsable --time=05:00:00 --job-name=v_glob_c2 --mem=200G --export=ALL,DATA="$DATA",RUN=v_glob_stagec,EXTRA="$BASE --init-from $R/v_glob --reuse-ae $R/v_glob $C" cluster/saidie/train_cm1.slurm)
je=$(sbatch --parsable --dependency=afterok:$jc --export=ALL,RUNS="v_glob_stagec" cluster/saidie/vit_eval.slurm)
echo "v_glob_stagec: $jc, eval $je"
