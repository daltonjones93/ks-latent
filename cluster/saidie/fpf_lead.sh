#!/bin/bash
# FPf (pre-registration 5h): channel 0 constrained to a closed short law in z0 (LatentLaw lead="z0", nu >= 0);
# Stage B with fpc_nested's autoencoder frozen, then Stage C (encoder free). Same recipe as fpc_nested(_stagec).
# ONE GPU, sequential.   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/fpf_lead.sh'
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
sub fpf_lead "$LAW --law-l1 1e-5 --nested-frac 0.5 --reuse-ae $R/fpc_nested --law-lead z0 --law-lead-nu-pos" afterany
sub fpf_lead_stagec "--init-from $R/fpf_lead --reuse-ae $R/fpf_lead $C --nested-frac 0.5"
