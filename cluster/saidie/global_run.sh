#!/bin/bash
# 7a: global coupling of the leading latent channels in the one-model stochastic system. Every arm shares the frozen
# nested-dropout autoencoder of fpc_nested_stagec (so all arms forecast the same latent), trains a stochastic CRPS
# conv-stencil propagator (Stage B, 4 noise channels, as full_v1_crps), then Stage C decoded CRPS with the autoencoder
# frozen. Arms differ only in the global module and the training window. 3 GPU lanes, sequential within a lane.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/global_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
R=$HOME/cm1_runs; AE=$R/fpc_nested_stagec
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8"
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 6 --c-freeze-ae"
ALL=""
lane() {   # arm specs "name|extra" run one after another on one GPU
  local prev=""
  for spec in "$@"; do
    local n=${spec%%|*} x=${spec#*|} dep
    dep=${prev:+--dependency=afterany:$prev}
    local jb; jb=$(sbatch --parsable --time=02:00:00 ${dep:-} --job-name=g_$n --mem=200G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE --reuse-ae $AE $x" cluster/saidie/train_cm1.slurm)
    prev=$(sbatch --parsable --time=02:00:00 --dependency=afterok:$jb --job-name=g_${n}_c --mem=200G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE --init-from $R/$n --reuse-ae $R/$n $C" cluster/saidie/train_cm1.slurm)
    echo "$n: B $jb, C $prev"; ALL="$ALL:$prev"
  done
}
W="--window-b 128"
lane "g_local48|--window-b 48"        "g_attn8|$W --global-kind attn --global-k 8"   "g_attn32|$W --global-kind attn --global-k 32"
lane "g_local128|$W"                  "g_mean8|$W --global-kind mean --global-k 8"     "g_attn8_s1|$W --global-kind attn --global-k 8 --seed 1"
lane "g_attn4|$W --global-kind attn --global-k 4"  "g_local128_s1|$W --seed 1"
je=$(sbatch --parsable --dependency=afterany$ALL cluster/saidie/global_eval.slurm)
echo "eval: $je"
