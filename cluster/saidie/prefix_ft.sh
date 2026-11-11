#!/bin/bash
# 5s: fine-tune the nested model after pruning its latent to the first M channels (the rest fixed at their training mean).
#   prop: propagator only (autoencoder frozen);  full: Stage C on the autoencoder and propagator together.
# Two sequential chains (one GPU each).   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/prefix_ft.sh'
#   STEPS=8000 MS="4 8 12 16" MODES="prop full"
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST"
R=$HOME/cm1_runs; STEPS=${STEPS:-8000}
for mode in ${MODES:-prop full}; do
  prev=${AFTER:-}
  for m in ${MS:-4 8 12 16}; do
    extra="--init-from $R/fpc_nested_stagec --stage-c-steps $STEPS --stage-c-k 45 --w-sfc-th 2 --nested-frac 0.5 --prefix-m $m"
    [ "$mode" = prop ] && extra="$extra --c-freeze-ae"
    dep=${prev:+--dependency=afterany:$prev}
    prev=$(sbatch --parsable ${dep:-} --mem=200G --export=ALL,DATA="$DATA",RUN=fpc_nested_p${m}_${mode},EXTRA="$BASE $extra" cluster/saidie/train_cm1.slurm)
    echo "p${m}_${mode}: $prev"
  done
done
