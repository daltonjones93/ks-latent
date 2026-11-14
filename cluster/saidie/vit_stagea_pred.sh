#!/bin/bash
# 8a: prediction term in Stage A (ViT 198 M windowed + original stochastic stencil propagator). A small throwaway stencil
# propagator is trained with the autoencoder and scored in physical space on a k-step decoded rollout (--w-pred-a, --pred-k-a).
# Stage C unfrozen exactly as u_win (cluster/saidie/vit_unfrozen.sh), so each arm differs from u_win only in Stage A.
#   a_k1     weight 1, one step ahead (1 min)
#   a_k4     weight 1, four steps ahead
#   a_k4w5   weight 5, four steps ahead (a heavier penalty)
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/vit_stagea_pred.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
R=$HOME/cm1_runs
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32"
A="--ae-kind vit --nested-frac 0.5 --steps-a 40000 --lr-a 3e-4 --warmup-a 1000 --vit-radius 2 --window-a 64"
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 5e-5"
ALL=""
for spec in "a_k1|1|1" "a_k4|1|4" "a_k4w5|5|4"; do
  IFS='|' read -r n w k <<< "$spec"
  jb=$(sbatch --parsable --time=04:00:00 --job-name=$n --mem=200G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE $A --w-pred-a $w --pred-k-a $k" cluster/saidie/train_cm1.slurm)
  jc=$(sbatch --parsable --time=05:00:00 --dependency=afterok:$jb --job-name=${n}_c --mem=200G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE --init-from $R/$n --reuse-ae $R/$n $C" cluster/saidie/train_cm1.slurm)
  echo "$n: A+B $jb, C $jc"; ALL="$ALL:$jc"
done
je=$(sbatch --parsable --dependency=afterany$ALL --export=ALL,OUT=v_eval_stagea,RUNS="a_k1_stagec a_k4_stagec a_k4w5_stagec u_win_stagec u_local48_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
