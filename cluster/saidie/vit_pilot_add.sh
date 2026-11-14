#!/bin/bash
# 8a pilot, additions (2026-11-13): the attention-propagator arm now uses SwiGLU FFNs (--prop-attn-swiglu), and two scaled-up
# conv column autoencoders (the original local model) with the ORIGINAL propagator (conv stencil, hidden 256, 3 layers,
# radius 1, 4 noise channels, CRPS; as full_v1_crps / g_local48): c_wide (hidden 3840, 2 mixing convs, reach 2 sites = the
# original design, 204 M) and c_deep (hidden 2048, 7 mixing convs, reach 7 sites ~ the windowed ViT's 8; 191 M). Same recipe
# as vit_pilot.sh. Resubmits v_win_attn / v_win_mlp after v_win's Stage C and the evaluation after everything.
#   ssh saidie 'cd ~/latent_DA && V_WIN_C=<id> LANE1=<id> LANE2=<id> PILOT_C=<id:id> bash -l cluster/saidie/vit_pilot_add.sh'
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
CONV="--ae-kind conv --nested-frac 0.5 --steps-a 40000 --lr-a 3e-4 --warmup-a 1000 --prop-body conv --prop-hidden 256"
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 6 --c-freeze-ae"
ALL=""
lane() {
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
LANE_AFTER="${V_WIN_C:?}" lane "v_win_attn|$WIN --prop-body attention --prop-attn-swiglu|v_win" "v_win_mlp|$WIN --prop-body mlp|v_win"
LANE_AFTER="${LANE1:?}" lane "c_wide|$CONV --hidden 3840 --n-mix 2|"
LANE_AFTER="${LANE2:?}" lane "c_deep|$CONV --hidden 2048 --n-mix 7 --window-a 64|"
je=$(sbatch --parsable --dependency=afterany$ALL:${PILOT_C:?} --export=ALL,RUNS="v_win_stagec v_win_attn_stagec v_win_mlp_stagec v_glob_stagec v_small_stagec c_wide_stagec c_deep_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
