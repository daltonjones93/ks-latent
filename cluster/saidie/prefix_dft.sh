#!/bin/bash
# 5t: (a) control: the UNPRUNED nested model given the same 8000 extra Stage C steps (whole model, and propagator only);
#     (b) distill-then-fine-tune: prune to m, 4000 Stage C steps against the unpruned model's rollouts (teacher), then the
#     same 8000 CM1 Stage C steps as 5s (whole model).   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/prefix_dft.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST"
R=$HOME/cm1_runs
C="--init-from $R/fpc_nested_stagec --stage-c-k 45 --w-sfc-th 2 --nested-frac 0.5"
sub() {  # run name, extra args, previous job
  local dep=${3:+--dependency=afterany:$3}
  sbatch --parsable ${dep:-} --mem=200G --export=ALL,DATA="$DATA",RUN=$1,EXTRA="$BASE $2" cluster/saidie/train_cm1.slurm
}
a=$(sub fpc_nested_ctl8k_full "$C --stage-c-steps 8000" ""); echo "ctl full: $a"
a=$(sub fpc_nested_ctl8k_prop "$C --stage-c-steps 8000 --c-freeze-ae" "$a"); echo "ctl prop: $a"
b=$(sub fpc_nested_p4_dft "$C --stage-c-steps 8000 --prefix-m 4 --distill-steps 4000" ""); echo "p4: $b"
b=$(sub fpc_nested_p8_dft "$C --stage-c-steps 8000 --prefix-m 8 --distill-steps 4000" "$b"); echo "p8: $b"
c=$(sub fpc_nested_p12_dft "$C --stage-c-steps 8000 --prefix-m 12 --distill-steps 4000" ""); echo "p12: $c"
c=$(sub fpc_nested_p16_dft "$C --stage-c-steps 8000 --prefix-m 16 --distill-steps 4000" "$c"); echo "p16: $c"
