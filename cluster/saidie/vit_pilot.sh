#!/bin/bash
# 8a pilot on the existing 121 storms (same split as the full run): ViT column autoencoders (SwiGLU, ratio 4, nested dropout
# 0.5, 32 channels) against the conv autoencoder, and three propagator bodies on the windowed ViT's latent. Each arm:
# Stage A (+ Stage B stochastic CRPS propagator, 4 noise channels) then Stage C decoded CRPS with the autoencoder frozen,
# the recipe of 7a's g_local48 (its conv-autoencoder control). Two GPU lanes (lane 2 after 7a's third lane: <= 4 GPUs).
#   ssh saidie 'cd ~/latent_DA && AFTER=<jobid> bash -l cluster/saidie/vit_pilot.sh'
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
VIT="--ae-kind vit --nested-frac 0.5 --steps-a 40000 --lr-a 3e-4 --warmup-a 1000"
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 6 --c-freeze-ae"
ALL=""
lane() {   # "name|extra|reuse_from" ...: Stage A+B (or B on a reused autoencoder), then Stage C
  local prev=${LANE_AFTER:-}
  for spec in "$@"; do
    IFS='|' read -r n x reuse <<< "$spec"
    local dep=${prev:+--dependency=afterany:$prev} jb
    local src=${reuse:+--reuse-ae $R/$reuse}
    jb=$(sbatch --parsable --time=${TB:-05:00:00} ${dep:-} --job-name=v_$n --mem=200G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE $x $src" cluster/saidie/train_cm1.slurm)
    prev=$(sbatch --parsable --time=${TC:-04:00:00} --dependency=afterok:$jb --job-name=v_${n}_c --mem=200G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE --init-from $R/$n --reuse-ae $R/$n $C" cluster/saidie/train_cm1.slurm)
    echo "$n: B $jb, C $prev"; ALL="$ALL:$prev"
  done
}
WIN="$VIT --vit-radius 2 --window-a 64"
LANE_AFTER="" lane "v_win|$WIN|" "v_win_attn|$WIN --prop-body attention|v_win" "v_win_mlp|$WIN --prop-body mlp|v_win"
LANE_AFTER="${AFTER:-}" lane "v_glob|$VIT --vit-radius 0 --window-a 200 --window-b 200|" \
     "v_small|$WIN --vit-dim 512 --vit-heads 8 --lr-a 5e-4|"
je=$(sbatch --parsable --dependency=afterany$ALL --export=ALL,RUNS="v_win_stagec v_win_attn_stagec v_win_mlp_stagec v_glob_stagec v_small_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
