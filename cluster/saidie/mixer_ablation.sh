#!/bin/bash
# 8b ablation (2026-11-16): does the RMSNorm + SwiGLU of the CM1 mixer matter? The same two grid arms as conv_scale.sh (fresh Stage A, 121 storms,
# Stage C unfrozen) with the mixer's norm and MLP replaced by LayerNorm + GELU MLP of the same parameter count (the Paper I/II block's details):
#   ab_h256_m128_gelu  against cs_h256_m128 (conv 2.6 M, mixer 0.75 M)      ab_h416_m512_gelu  against cs_h416_m512 (conv 5.2 M, mixer ~12 M)
# Tight time limits (measured on the SwiGLU arms: A+B 32 / 51 min, C 28 / 32 min, evaluation 13 min), 96 GB host memory (121 storms are read into memory).
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/mixer_ablation.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
R=$HOME/cm1_runs
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mixer-variant gelu_ln"
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"
ALL=""
for spec in "ab_h256_m128_gelu|256|128|00:45:00|00:40:00" "ab_h416_m512_gelu|416|512|01:10:00|00:45:00"; do
  IFS='|' read -r n h w tab tc <<< "$spec"
  jb=$(sbatch --parsable --time=$tab --job-name=$n --mem=96G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE --ae-kind conv --hidden $h --nested-frac 0.5 --steps-a 30000 --prop-body mixer --prop-hidden $w --window-b 48" cluster/saidie/train_cm1.slurm)
  jc=$(sbatch --parsable --time=$tc --dependency=afterok:$jb --job-name=${n}_c --mem=96G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE --ae-kind conv --hidden $h --prop-body mixer --prop-hidden $w --init-from $R/$n --reuse-ae $R/$n $C" cluster/saidie/train_cm1.slurm)
  echo "$n: A+B $jb, C $jc"; ALL="$ALL:$jc"
done
je=$(sbatch --parsable --time=00:25:00 --dependency=afterany$ALL --export=ALL,OUT=v_eval_ablation,RUNS="ab_h256_m128_gelu_stagec ab_h416_m512_gelu_stagec cs_h256_m128_stagec cs_h416_m512_stagec um_local_stagec u_local48_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
