#!/bin/bash
# 8a: the Paper I/II local_mlp block as the CM1 propagator (user: revisit the propagator structure; option 1). Same frozen
# nested-dropout conv autoencoder (fpc_nested_stagec) and recipe as g_local48 (the control: original conv stencil, 0.43 M):
# Stage B stochastic CRPS propagator (4 noise channels, as an input), Stage C decoded CRPS with the autoencoder frozen.
#   m128    mixer, width 128, 3 blocks radius 1 (reach 3 sites), SwiGLU x4: 0.75 M   (1.7x the control; the main arm)
#   m256    mixer, width 256: 3.0 M                                                    (capacity check)
#   mlp256  dense SwiGLU MLP on the 7-site neighbourhood, width 256: 1.65 M            (the other structure, for reference)
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/mixer_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
R=$HOME/cm1_runs; AE=$R/fpc_nested_stagec
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8"
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 6 --c-freeze-ae"
ALL=""
for spec in "m128|--prop-body mixer --prop-hidden 128" "m256|--prop-body mixer --prop-hidden 256" "mlp256|--prop-body mlp --prop-hidden 256"; do
  n=${spec%%|*}; x=${spec#*|}
  jb=$(sbatch --parsable --time=02:00:00 --job-name=$n --mem=200G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE --reuse-ae $AE --window-b 48 $x" cluster/saidie/train_cm1.slurm)
  jc=$(sbatch --parsable --time=02:00:00 --dependency=afterok:$jb --job-name=${n}_c --mem=200G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE --init-from $R/$n --reuse-ae $R/$n $C" cluster/saidie/train_cm1.slurm)
  echo "$n: B $jb, C $jc"; ALL="$ALL:$jc"
done
je=$(sbatch --parsable --dependency=afterany$ALL --export=ALL,OUT=v_eval_mixer,RUNS="m128_stagec m256_stagec mlp256_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
