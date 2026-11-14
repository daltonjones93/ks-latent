#!/bin/bash
# 8d (user: double the conv autoencoder and make the mixer much larger: does CRPS / MSE improve?). A 2x2 grid with a FRESH Stage A each
# (nested dropout 0.5, 30k steps) on the 121-storm data, stochastic mixer propagator (4 noise channels, CRPS), Stage C unfrozen
# (lr 1e-4, batch 4, as um_local):   conv hidden 256 (2.6 M) or 416 (5.2 M, 2x)  x  mixer width 128 (0.75 M) or 512 (~12 M).
# cs_h256_m128 is the matched control (fresh Stage A; um_local reused the Stage-C-adapted autoencoder of fpc_nested_stagec).
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/conv_scale.sh'
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
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"
ALL=""
for spec in "cs_h256_m128|256|128" "cs_h416_m128|416|128" "cs_h256_m512|256|512" "cs_h416_m512|416|512"; do
  IFS='|' read -r n h w <<< "$spec"
  jb=$(sbatch --parsable --time=01:15:00 --job-name=$n --mem=200G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE --ae-kind conv --hidden $h --nested-frac 0.5 --steps-a 30000 --prop-body mixer --prop-hidden $w --window-b 48" cluster/saidie/train_cm1.slurm)
  jc=$(sbatch --parsable --time=01:15:00 --dependency=afterok:$jb --job-name=${n}_c --mem=200G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE --ae-kind conv --hidden $h --prop-body mixer --prop-hidden $w --init-from $R/$n --reuse-ae $R/$n $C" cluster/saidie/train_cm1.slurm)
  echo "$n: A+B $jb, C $jc"; ALL="$ALL:$jc"
done
je=$(sbatch --parsable --time=00:45:00 --dependency=afterany$ALL --export=ALL,OUT=v_eval_convscale,RUNS="cs_h256_m128_stagec cs_h416_m128_stagec cs_h256_m512_stagec cs_h416_m512_stagec um_local_stagec u_local48_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
