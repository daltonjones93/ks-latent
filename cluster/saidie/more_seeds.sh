#!/bin/bash
# 10b (user: "averaging independently trained models is probably the best idea... determine how many to use"): F6 seeds 3, 4, 5, so the
# multi-model pool can be studied at 1-6 models (seeds 0 = F6, 1-2 from audit_runs.sh). F6's recipe exactly; limits ~1.3x F6's times.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/more_seeds.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C+$HOME/cm1_cache_C2+$HOME/cm1_cache_D"
R=$HOME/cm1_runs
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mmap --steps-b 40000 --mixer-variant gelu_ln"
CA="--ae-kind conv --hidden 416 --prop-body mixer --prop-hidden 512 --window-b 48"
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"
for sd in 3 4 5; do
  n=F6s${sd}
  ab=$(sbatch --parsable --time=02:55:00 --job-name=$n --mem=56G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE $CA --nested-frac 0.5 --steps-a 60000 --w-pred-a 1 --pred-k-a 1 --seed $sd" cluster/saidie/train_cm1.slurm)
  c=$(sbatch --parsable --time=01:55:00 --dependency=afterok:$ab --job-name=${n}_stagec --mem=56G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE $CA --init-from $R/$n --reuse-ae $R/$n $C --seed $sd" cluster/saidie/train_cm1.slurm)
  echo "$n: A+B $ab, C $c"
done
