#!/bin/bash
# 8c: full training runs on the large dataset (ens_A + ens_A2 + ens_B + the new ens_C storms copied from Cashew), the same 6 validation and 22
# test storms as every earlier run. Memory-mapped caches (--mmap). All Stage C with the autoencoder unfrozen (lr 5e-5 for the ViTs,
# 1e-4 for the conv model, batch 4, nested dropout 0.5), stochastic CRPS propagators (4 noise channels).
#   F1  ViT 75 M (736 wide, 23 heads, windowed radius 2, SwiGLU x4) + dense SwiGLU MLP propagator (1.65 M) + Stage A prediction term
#       (weight 1, 1 step ahead): the best ViT recipe of 8a
#   F2  the same without the Stage A prediction term (isolates it)
#   F3  conv 2.6 M autoencoder + mixer propagator (0.75 M): the best CRPS so far on 121 storms, as the small-model control on the same data
# Steps are doubled against the 121-storm pilot (Stage A 80k, Stage B 40k, Stage C 30k; F3: Stage A 60k).
#   ssh saidie 'cd ~/latent_DA && CACHES="cache:..." bash -l cluster/saidie/full_data_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA=${CACHES:-"cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C"}
R=$HOME/cm1_runs
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mmap --steps-b 40000"
VIT="--ae-kind vit --vit-dim 736 --vit-heads 23 --vit-radius 2 --window-a 64 --nested-frac 0.5 --steps-a 80000 --lr-a 3e-4 --warmup-a 1000"
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5"
ALL=""
arm() {   # name, stage A+B extra args, stage C lr, AE args for the Stage C reuse
  local n=$1 x=$2 lr=$3 ae=$4 jb jc
  jb=$(sbatch --parsable --time=02:30:00 --job-name=$n --mem=128G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE $x" cluster/saidie/train_cm1.slurm)
  jc=$(sbatch --parsable --time=01:45:00 --dependency=afterok:$jb --job-name=${n}_c --mem=128G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE --init-from $R/$n --reuse-ae $R/$n $C --stage-c-lr $lr" cluster/saidie/train_cm1.slurm)
  echo "$n: A+B $jb, C $jc"; ALL="$ALL:$jc"
}
arm F1_vit75_mlp_pred "$VIT --prop-body mlp --prop-hidden 256 --w-pred-a 1 --pred-k-a 1" 5e-5
arm F2_vit75_mlp      "$VIT --prop-body mlp --prop-hidden 256" 5e-5
arm F3_conv_mixer     "--ae-kind conv --nested-frac 0.5 --steps-a 60000 --prop-body mixer --prop-hidden 128 --window-b 48" 1e-4
je=$(sbatch --parsable --time=01:00:00 --dependency=afterany$ALL --export=ALL,OUT=v_eval_full,RUNS="F1_vit75_mlp_pred_stagec F2_vit75_mlp_stagec F3_conv_mixer_stagec um_local_stagec u_win_mlp_stagec g_local48_stagec full_v1_stagec_crps",EXTRA_EVAL="--ref-norm-run \$HOME/cm1_runs/um_local_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
