#!/bin/bash
# SM2 / SM3: probabilistic objectives on decoded samples (full-run data and split, as full_run.sh).
#   sm2a: Stage C of the stochastic model (full_v1_crps) with CRPS + energy score (w 1)
#   sm2b: a new stochastic propagator with 8 noise channels on full_v1's autoencoder, then CRPS + energy
#   sm2c: as sm2a with 30 000 Stage C steps
#   sm3 : sm2a + spectral term (w 0.1)
# All chained on one GPU after AFTER.   ssh saidie 'cd ~/latent_DA && AFTER=<job> bash -l cluster/saidie/sm2_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
SPLIT="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --eval-samples 8"
R=$HOME/cm1_runs
C="--stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 6"
dep=${AFTER:+--dependency=afterany:$AFTER}
LAST=""
sub() {   # name extra [dependency override]
  LAST=$(sbatch --parsable ${3:-${dep:-}} --mem=200G --time=04:00:00 --export=ALL,DATA="$DATA",RUN=$1,EXTRA="$SPLIT $2" cluster/saidie/train_cm1.slurm)
  echo "$1: $LAST"; dep="--dependency=afterany:$LAST"
}
sub sm2a_energy "--init-from $R/full_v1_crps --reuse-ae $R/full_v1_crps --stage-c-steps 15000 $C --stage-c-energy 1.0"
sub sm2b_noise8 "--reuse-ae $R/full_v1 --noise-channels 8 --crps-samples 4"
sub sm2b_noise8_stagec "--init-from $R/sm2b_noise8 --reuse-ae $R/sm2b_noise8 --stage-c-steps 15000 $C --stage-c-energy 1.0" "--dependency=afterok:$LAST"
sub sm2c_long "--init-from $R/full_v1_crps --reuse-ae $R/full_v1_crps --stage-c-steps 30000 $C --stage-c-energy 1.0"
sub sm3_spectral "--init-from $R/full_v1_crps --reuse-ae $R/full_v1_crps --stage-c-steps 15000 $C --stage-c-energy 1.0 --stage-c-spectral 0.1"
