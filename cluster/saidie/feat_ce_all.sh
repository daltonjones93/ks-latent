#!/bin/bash
# 9e (user: do the presence cross-entropy "for liquid and all the other variables in CM1... start with the separate model first"): one arm per
# variable, each = the deterministic F7 (Stage B) -> F7's Stage C with MSE + 1.0 x the cross-entropy of that variable's features
# (cell_losses.FEATURES / FEATURE_GROUPS; calibration in scripts/cm1/feature_calibration_all.py):
#   u: u' > +10 and < -10 m/s     th: theta' < -2 K and > +5 K     qv: qv' > +1 and < -3 g/kg     prs: p' > +3 and < -1 hPa
#   qliq: liquid > 0.5 g/kg        qice: ice > 1 g/kg               (w: w > 2 and < -1.5 m/s is F12_det_ce, cluster/saidie/feat_ce.sh)
# Last state kept. A joint arm (all variables in one model) follows once these are read.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/feat_ce_all.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
R=$HOME/cm1_runs; S7=$R/F7_conv416_mixer512_det
D7="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C"
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4 --c-select-last --c-feat-weight 1"
CA="--ae-kind conv --hidden 416 --prop-body mixer --prop-hidden 512 --window-b 48"
COMMON="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --tf32 --mmap --steps-b 40000"
ALL=""; RUNS=""
for v in u th qv prs qliq qice; do
  n=F13_ce_$v
  j=$(sbatch --parsable --time=01:35:00 --job-name=$n --mem=56G --export=ALL,DATA="$D7",RUN=${n}_stagec,EXTRA="$COMMON $CA --init-from $S7 --reuse-ae $S7 $C --c-feat-names $v" cluster/saidie/train_cm1.slurm)
  echo "$n: $j"; ALL="$ALL:$j"; RUNS="$RUNS ${n}_stagec"
done
je=$(sbatch --parsable --time=00:40:00 --dependency=afterany$ALL:${W_ARM:-9615} --export=ALL,OUT=v_eval_feat_all,RUNS="${RUNS# } F12_det_ce_stagec F7_conv416_mixer512_det_stagec F6_conv416_mixer512_gelu_pred_all_stagec g_local48_stagec",EXTRA_EVAL="--ref-norm-run ${HOME}/cm1_runs/um_local_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
