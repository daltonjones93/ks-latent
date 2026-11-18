#!/bin/bash
# 9a (user: "sure do that on F7"): the cell-formation terms (ks_latent/cm1/cell_losses.py) in the Stage C of the deterministic F7. Each arm starts
# from F7's Stage B (job 9544: F4's autoencoder + the width-512 mixer without noise, MSE rollout loss) and runs the same Stage C as F7 (30k steps, k 45,
# batch 4, lr 1e-4, autoencoder unfrozen) with the decoded MSE PLUS one or all of:
#   tw    3.0 x threshold-weighted absolute error (w above 2 m/s, below -2; liquid and ice above 0.1 g/kg; smooth chaining function)
#   pool  3.0 x neighbourhood-pooled exceedance absolute error (2, 4, 8, 16 km)
#   lp    2.0 x squared error after a 10 km Gaussian low-pass along x
# Weights are not tuned: chosen so each term is of the order of half the decoded MSE (~0.14; the terms measure about 0.02-0.07 per unit weight).
# The last state is kept (--c-select-last): the best-decoded-MSE state would pick the blurriest.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/det_f7_cells.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C"
R=$HOME/cm1_runs; SRC=$R/F7_conv416_mixer512_det
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --tf32 --mmap --steps-b 40000"
CA="--ae-kind conv --hidden 416 --prop-body mixer --prop-hidden 512 --window-b 48"
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4 --c-select-last"
DEP=${AFTER:-9544}
ALL=""
for spec in "F7_tw|--c-tw-weight 3" "F7_pool|--c-pool-weight 3" "F7_lp|--c-lp-weight 2" "F7_all|--c-tw-weight 3 --c-pool-weight 3 --c-lp-weight 2"; do
  n=${spec%%|*}; x=${spec#*|}
  jc=$(sbatch --parsable --time=01:00:00 --dependency=afterok:$DEP --job-name=$n --mem=48G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE $CA --init-from $SRC --reuse-ae $SRC $C $x" cluster/saidie/train_cm1.slurm)
  echo "$n: $jc"; ALL="$ALL:$jc"
done
je=$(sbatch --parsable --time=00:30:00 --dependency=afterany$ALL:${EVAL_AFTER:-9545} --export=ALL,OUT=v_eval_cells,RUNS="F7_tw_stagec F7_pool_stagec F7_lp_stagec F7_all_stagec F7_conv416_mixer512_det_stagec F4_conv416_mixer512_pred_stagec g_local48_stagec",EXTRA_EVAL="--ref-norm-run \$HOME/cm1_runs/um_local_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
