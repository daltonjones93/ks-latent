#!/bin/bash
# SM1: representation capacity. The full-run recipe (all three ensembles, same split as full_run.sh) with
# 64 latent channels, and with 1 km sites (patch 4; physical reach unchanged), each followed by Stage C.
# All four jobs chained on one GPU after AFTER.
#   ssh saidie 'cd ~/latent_DA && AFTER=<job> bash -l cluster/saidie/sm1_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
SPLIT="--k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST"
dep=${AFTER:+--dependency=afterany:$AFTER}
for spec in "sm1_c64|--latent-channels 64" "sm1_patch4|--latent-channels 32 --patch 4"; do
  run=${spec%%|*}; extra=${spec#*|}
  j1=$(sbatch --parsable ${dep:-} --mem=200G --time=03:00:00 --export=ALL,DATA="$DATA",RUN=$run,EXTRA="$SPLIT $extra" cluster/saidie/train_cm1.slurm)
  R=$HOME/cm1_runs/$run
  j2=$(sbatch --parsable --dependency=afterok:$j1 --mem=200G --time=03:00:00 --export=ALL,DATA="$DATA",RUN=${run}_stagec,EXTRA="$SPLIT $extra --init-from $R --reuse-ae $R --stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2" cluster/saidie/train_cm1.slurm)
  echo "$run $j1, ${run}_stagec $j2"
  dep="--dependency=afterany:$j2"
done
