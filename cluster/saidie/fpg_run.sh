#!/bin/bash
# FPg (pre-registration 5i): closed PDE head on the first n channels with a representation loss against the dense law.
# Stage B on the frozen fpc_nested autoencoder, then Stage C. n = 4, w in {0.1, 1}. ONE GPU, sequential.
# AFTER=<jobid> to start after a running job.   ssh saidie 'cd ~/latent_DA && AFTER=9161 bash -l cluster/saidie/fpg_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST"
LAW="--prop-body law --law-res-weight 0.1 --law-cap 5"   # cap: 8999 blew up without it (see the plan)
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2"
R=$HOME/cm1_runs
prev=${AFTER:-}
sub() {   # name, extra args, dependency type
  local dep=${prev:+--dependency=${3:-afterok}:$prev}
  prev=$(sbatch --parsable ${dep:-} --mem=200G --export=ALL,DATA="$DATA",RUN=$1,EXTRA="$BASE $2" cluster/saidie/train_cm1.slurm)
  echo "$1: $prev"
}
for w in 0.1 1; do
  t=fpg_n4_w${w/./}
  sub $t "$LAW --law-l1 1e-5 --nested-frac 0.5 --reuse-ae $R/fpc_nested --law-head-n 4 --law-head-weight $w" afterany
  sub ${t}_stagec "--init-from $R/$t --reuse-ae $R/$t $C --nested-frac 0.5"
done
