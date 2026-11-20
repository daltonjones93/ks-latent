#!/bin/bash
# 9d (user: train on updraft / downdraft presence masks with cross-entropy instead of MSE, "maybe we would get sharper descriptions"):
# updraft = w > 2 m/s, downdraft = w < -1.5 m/s (calibrated: scripts/cm1/feature_calibration.py, ~1.6% of the domain each, 5-8x the
# reconstruction error of w); the decoded w becomes a probability through a sigmoid of width 0.5 m/s; binary cross-entropy, no class reweighting.
#   F12_det_ce      deterministic F7 (Stage B) -> F7's Stage C: MSE + 1.0 x cross-entropy
#   F12_det_ceonly  deterministic F7 (Stage B) -> F7's Stage C with the w channels' MSE removed: the cross-entropy replaces it for w
#                   (the literal "instead of MSE"; the other variables keep their MSE)
#   F12_sto_ce      stochastic F6 (Stage B) -> F6's Stage C: fair CRPS + 1.0 x log score of the ensemble probability (proper)
# The last state is kept (selection on the validation MSE / CRPS would favour the state the term moves away from). Controls: the existing
# F7 and F6 Stage C runs. Limits: F7's Stage C took 73 min, F6's 90 min (F11 93 min).
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/feat_ce.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
R=$HOME/cm1_runs
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4 --c-select-last --c-feat-weight 1"
CA="--ae-kind conv --hidden 416 --prop-body mixer --prop-hidden 512 --window-b 48"
COMMON="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --tf32 --mmap --steps-b 40000"
# deterministic (F7: 325 storms, SwiGLU mixer, no noise)
D7="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C"; S7=$R/F7_conv416_mixer512_det
# stochastic (F6: 573 storms, GELU mixer, 4 noise channels, 4 CRPS samples)
D6="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C+$HOME/cm1_cache_C2+$HOME/cm1_cache_D"; S6=$R/F6_conv416_mixer512_gelu_pred_all
ALL=""
sub() { # name time data extra
  local j; j=$(sbatch --parsable --time=$2 --job-name=$1 --mem=56G --export=ALL,DATA="$3",RUN=${1}_stagec,EXTRA="$4" cluster/saidie/train_cm1.slurm); echo "$1: $j"; ALL="$ALL:$j"; }
sub F12_det_ce     01:35:00 "$D7" "$COMMON $CA --init-from $S7 --reuse-ae $S7 $C"
sub F12_det_ceonly 01:35:00 "$D7" "$COMMON $CA --init-from $S7 --reuse-ae $S7 $C --c-w-mse-scale 0"
sub F12_sto_ce     02:00:00 "$D6" "$COMMON $CA --noise-channels 4 --crps-samples 4 --eval-samples 8 --mixer-variant gelu_ln --init-from $S6 --reuse-ae $S6 $C --stage-c-crps 4"
je=$(sbatch --parsable --time=00:30:00 --dependency=afterany$ALL --export=ALL,OUT=v_eval_feat,RUNS="F12_det_ce_stagec F12_det_ceonly_stagec F12_sto_ce_stagec F7_conv416_mixer512_det_stagec F7_all_stagec F6_conv416_mixer512_gelu_pred_all_stagec g_local48_stagec",EXTRA_EVAL="--ref-norm-run ${HOME}/cm1_runs/um_local_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
