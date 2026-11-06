#!/bin/bash
# M1a: windowed-attention propagator against the convolutional one, on full_v1's autoencoder (full data and
# split as full_run.sh), each Stage B then Stage C (as full_v1 -> full_v1_stagec):
#   m1a_attn128_s0 / _s1  attention, width 128 (0.41 M parameters, matched to the conv body's 0.43 M), seeds 0, 1
#   m1a_conv_s1           conv, seed 1 (seed noise: conv seed 0 is full_v1 / full_v1_stagec)
#   m1a_attn256_s0        attention, width 256 (1.6 M parameters): capacity check
# All chained on one GPU after AFTER.   ssh saidie 'cd ~/latent_DA && AFTER=<job> bash -l cluster/saidie/m1a_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
SPLIT="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST"
R=$HOME/cm1_runs
dep=${AFTER:+--dependency=afterany:$AFTER}
for spec in "m1a_attn128_s0|--prop-body attention --prop-hidden 128 --seed 0" "m1a_attn128_s1|--prop-body attention --prop-hidden 128 --seed 1" \
            "m1a_conv_s1|--seed 1" "m1a_attn256_s0|--prop-body attention --prop-hidden 256 --seed 0"; do
  run=${spec%%|*}; extra=${spec#*|}
  j1=$(sbatch --parsable ${dep:-} --mem=200G --time=03:00:00 --export=ALL,DATA="$DATA",RUN=$run,EXTRA="$SPLIT $extra --reuse-ae $R/full_v1" cluster/saidie/train_cm1.slurm)
  j2=$(sbatch --parsable --dependency=afterok:$j1 --mem=200G --time=03:00:00 --export=ALL,DATA="$DATA",RUN=${run}_stagec,EXTRA="$SPLIT $extra --init-from $R/$run --reuse-ae $R/$run --stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2" cluster/saidie/train_cm1.slurm)
  echo "$run $j1, ${run}_stagec $j2"
  dep="--dependency=afterany:$j2"
done
