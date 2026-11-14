#!/bin/bash
# 8a: 75 M-parameter windowed ViT autoencoder (vit_dim 736, 23 heads of 32, 4+4 blocks, SwiGLU x4; 74.5 M), replacing the
# 198 M model for the Stage A prediction-term arms (user: 200 M is probably too much). Original stochastic stencil propagator,
# Stage C unfrozen (as u_win). w75 is the matched control (no prediction term); p75_* add the Stage A prediction term:
#   p75_k1 (weight 1, 1 step), p75_k4 (1, 4), p75_k4w5 (5, 4), p75_k8 (1, 8), p75_k16 (1, 16; 4 windows carry the extra frames)
# Three lanes of two arms.   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/vit75.sh'
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
A="--ae-kind vit --vit-dim 736 --vit-heads 23 --nested-frac 0.5 --steps-a 40000 --lr-a 3e-4 --warmup-a 1000 --vit-radius 2 --window-a 64"
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 5e-5"
ALL=""
lane() {
  local prev=""
  for spec in "$@"; do
    IFS='|' read -r n w k pb <<< "$spec"
    local dep=${prev:+--dependency=afterany:$prev} jb ex=""
    [ "$w" != "0" ] && ex="--w-pred-a $w --pred-k-a $k --pred-batch-a $pb"
    jb=$(sbatch --parsable --time=04:00:00 ${dep:-} --job-name=$n --mem=200G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE $A $ex" cluster/saidie/train_cm1.slurm)
    prev=$(sbatch --parsable --time=04:00:00 --dependency=afterok:$jb --job-name=${n}_c --mem=200G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE --init-from $R/$n --reuse-ae $R/$n $C" cluster/saidie/train_cm1.slurm)
    echo "$n: A+B $jb, C $prev"; ALL="$ALL:$prev"
  done
}
lane "w75|0|0|0" "p75_k8|1|8|8"
lane "p75_k1|1|1|8" "p75_k16|1|16|4"
lane "p75_k4|1|4|8" "p75_k4w5|5|4|8"
je=$(sbatch --parsable --dependency=afterany$ALL --export=ALL,OUT=v_eval_75,RUNS="w75_stagec p75_k1_stagec p75_k4_stagec p75_k4w5_stagec p75_k8_stagec p75_k16_stagec u_win_stagec u_local48_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
