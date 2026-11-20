#!/bin/bash
# 9f (user: "figure out how to fix F12, maybe reweight so we correctly predict the presence... launch F13 as the fix"). The F13_ce_* names are
# the 9e per-variable arms; the fix arms are F13_fix_*. Diagnosis (docs/fpe_report/data/soft_cal.json): F12_det_ce's probabilities are roughly
# calibrated, but a deterministic cross-entropy model draws a feature only where p > 0.5; the threshold that unbiases the area is p ~ 0.25 /
# 0.15 / 0.10 at +15 / +30 / +60 min for updrafts (0.25 / 0.20 / 0.10 downdrafts), and it is also the CSI-maximizing threshold. The class
# weight that moves the decision there is (1 - t) / t ~ 3 / 6 / 9 ~ 1 + lead (min) / 7.
#   F13_fix_w    MSE + 1.0 x weighted cross-entropy of the w features, class weight 1 + k / 7 per rollout step k (calibrated)
#   F13_fix_w2   the same with twice the slope, 1 + 2k / 7 (bracket: the calibration came from an unweighted model, which retraining changes)
#   F13_fix_csi  MSE + 0.3 x soft CSI of the w features (Lagerquist and Ebert-Uphoff 2022; no weight to calibrate; weight 0.3 untuned)
# Each from F7's Stage B with F7's Stage C (deterministic, 30k steps), last state. Benchmark to beat: F7 with a post-hoc lower threshold
# (w > 1.3 m/s counted as an updraft at +30 min: bias 0.98, CSI 0.331 against 0.286 at 2 m/s).
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/feat_fix.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
R=$HOME/cm1_runs; S7=$R/F7_conv416_mixer512_det
D7="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C"
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4 --c-select-last --c-feat-names w"
CA="--ae-kind conv --hidden 416 --prop-body mixer --prop-hidden 512 --window-b 48"
COMMON="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --tf32 --mmap --steps-b 40000"
ALL=""
sub() { local j; j=$(sbatch --parsable --time=01:35:00 --job-name=$1 --mem=56G --export=ALL,DATA="$D7",RUN=${1}_stagec,EXTRA="$COMMON $CA --init-from $S7 --reuse-ae $S7 $C $2" cluster/saidie/train_cm1.slurm); echo "$1: $j"; ALL="$ALL:$j"; }
sub F13_fix_w   "--c-feat-weight 1 --c-feat-pos-slope 0.142857"
sub F13_fix_w2  "--c-feat-weight 1 --c-feat-pos-slope 0.285714"
sub F13_fix_csi "--c-feat-weight 0.3 --c-feat-kind csi"
je=$(sbatch --parsable --time=00:35:00 --dependency=afterany$ALL --export=ALL,OUT=v_eval_fix,RUNS="F13_fix_w_stagec F13_fix_w2_stagec F13_fix_csi_stagec F12_det_ce_stagec F7_conv416_mixer512_det_stagec F7_all_stagec F6_conv416_mixer512_gelu_pred_all_stagec g_local48_stagec",EXTRA_EVAL="--ref-norm-run ${HOME}/cm1_runs/um_local_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
