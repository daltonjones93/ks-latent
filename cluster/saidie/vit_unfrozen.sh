#!/bin/bash
# 8a follow-up (2026-11-14): the pilot froze the autoencoder in Stage C (copied from 7a), but the original best model
# (full_v1_stagec_crps) ran Stage C with the autoencoder free, so the latent adapts to be forecastable. Here Stage C is rerun
# with the autoencoder UNFROZEN (nested dropout 0.5 kept so the channel ordering survives), from the pilot's Stage B outputs:
#   u_win      ViT 198 M + original stencil          (v_win)
#   u_win_mlp  ViT 198 M + dense SwiGLU MLP          (v_win_mlp)
#   u_wide     conv 204 M + original stencil         (c_wide)
#   u_local48  conv 2.6 M + original stencil         (the control: fpc_nested_stagec's autoencoder)
# Autoencoder learning rate 5e-5 for the 200 M models, 1e-4 for the control; batch 4 (the encoder is now in the graph).
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/vit_unfrozen.sh'
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
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5"
ALL=""
lane() {
  local prev=""
  for spec in "$@"; do
    IFS='|' read -r n src lr <<< "$spec"
    local dep=${prev:+--dependency=afterany:$prev}
    prev=$(sbatch --parsable --time=05:00:00 ${dep:-} --job-name=$n --mem=200G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE --init-from $R/$src --reuse-ae $R/$src $C --stage-c-lr $lr" cluster/saidie/train_cm1.slurm)
    echo "$n: $prev"; ALL="$ALL:$prev"
  done
}
lane "u_win|v_win|5e-5" "u_win_mlp|v_win_mlp|5e-5"
lane "u_wide|c_wide|5e-5" "u_local48|g_local48|1e-4"
je=$(sbatch --parsable --dependency=afterany$ALL --export=ALL,OUT=v_eval_unfrozen,RUNS="u_win_stagec u_win_mlp_stagec u_wide_stagec u_local48_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
