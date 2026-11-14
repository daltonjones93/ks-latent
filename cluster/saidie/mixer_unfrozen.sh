#!/bin/bash
# 8b follow-up: the Paper I/II mixer propagator (width 128, 0.75 M) with Stage C unfrozen, on three autoencoders (reusing their
# Stage A): um_local (conv 2.6 M, fpc_nested_stagec), um_w75 (ViT 75 M from w75), um_win (ViT 198 M from v_win). Stage B stochastic CRPS
# (4 noise channels), Stage C decoded CRPS with the autoencoder free (batch 4, nested dropout 0.5; lr 1e-4 for the conv model, 5e-5
# for the ViTs), as the u_* arms.   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/mixer_unfrozen.sh'
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
P="--prop-body mixer --prop-hidden 128 --window-b 48"
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5"
ALL=""
for spec in "um_local|fpc_nested_stagec|--ae-kind conv|1e-4" "um_w75|w75|--ae-kind vit --vit-dim 736 --vit-heads 23 --vit-radius 2|5e-5" "um_win|v_win|--ae-kind vit --vit-radius 2|5e-5"; do
  IFS='|' read -r n src aeargs lr <<< "$spec"
  jb=$(sbatch --parsable --time=00:50:00 --job-name=$n --mem=200G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE $aeargs $P --reuse-ae $R/$src" cluster/saidie/train_cm1.slurm)
  jc=$(sbatch --parsable --time=01:15:00 --dependency=afterok:$jb --job-name=${n}_c --mem=200G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE --init-from $R/$n --reuse-ae $R/$n $C --stage-c-lr $lr" cluster/saidie/train_cm1.slurm)
  echo "$n: B $jb, C $jc"; ALL="$ALL:$jc"
done
je=$(sbatch --parsable --time=00:45:00 --dependency=afterany$ALL --export=ALL,OUT=v_eval_mixer_unfrozen,RUNS="um_local_stagec um_w75_stagec um_win_stagec u_win_mlp_stagec u_local48_stagec m128_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
