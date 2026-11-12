#!/bin/bash
# 5v launcher: three chains (one GPU each).  ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/da_prune.sh'
set -euo pipefail
cd "$HOME/latent_DA"
sub() { local dep=${3:+--dependency=afterany:$3}; sbatch --parsable ${dep:-} --export=ALL,RUN=$1,LABEL=$2 cluster/saidie/da_prune.slurm; }
a=$(sub fpc_nested_stagec n32 ""); echo "n32: $a"; a=$(sub fpc_nested_p4_dft p4 "$a"); echo "p4: $a"
b=$(sub fpc_nested_p16_dft p16 ""); echo "p16: $b"; b=$(sub fpc_nested_p12_dft p12 "$b"); echo "p12: $b"
c=$(sub fpc_nested_p8_dft p8 ""); echo "p8: $c"
