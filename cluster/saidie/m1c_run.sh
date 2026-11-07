#!/bin/bash
# M1c: capacity control for M1a. A conv propagator 512 wide (1.6 M parameters, matched to m1a_attn256's
# 1.60 M) on full_v1's autoencoder, Stage B then Stage C exactly as m1a_run.sh. If it matches attn256 at
# 5 min, M1a's gain is capacity, not attention.
#   ssh saidie 'cd ~/latent_DA && AFTER=<job> bash -l cluster/saidie/m1c_run.sh'
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
run=m1c_conv512_s0; extra="--prop-hidden 512 --seed 0"
j1=$(sbatch --parsable ${dep:-} --mem=200G --time=03:00:00 --export=ALL,DATA="$DATA",RUN=$run,EXTRA="$SPLIT $extra --reuse-ae $R/full_v1" cluster/saidie/train_cm1.slurm)
j2=$(sbatch --parsable --dependency=afterok:$j1 --mem=200G --time=03:00:00 --export=ALL,DATA="$DATA",RUN=${run}_stagec,EXTRA="$SPLIT $extra --init-from $R/$run --reuse-ae $R/$run --stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2" cluster/saidie/train_cm1.slurm)
echo "$run $j1, ${run}_stagec $j2"
