#!/bin/bash
# 8g (user: how much mean error does F4 give up for its structure?): F7 = the deterministic twin of F4. F4's autoencoder (Stage A with the
# prediction term, ae.pt of F4_conv416_mixer512_pred) is reused; the mixer propagator (width 512, SwiGLU as F4) has no noise channels and is trained
# with the MSE rollout loss in Stage B (latent) and the decoded MSE loss in Stage C (autoencoder unfrozen, lr 1e-4, batch 4, nested 0.5), on the same 325
# storms as F4. Compare with F4 on RMSE of the (sample-mean) forecast, spectra and sections; CRPS of the deterministic model is its mean absolute error.
# Limits from F4 (B 75 min and C 86 min with 4 CRPS samples; without samples about a third of that) with margin.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/det_f4.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C"
R=$HOME/cm1_runs; n=F7_conv416_mixer512_det
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --tf32 --mmap --steps-b 40000"
CA="--ae-kind conv --hidden 416 --prop-body mixer --prop-hidden 512 --window-b 48"
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"
jb=$(sbatch --parsable --time=01:10:00 --job-name=F7_B --mem=48G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE $CA --nested-frac 0.5 --reuse-ae $R/F4_conv416_mixer512_pred" cluster/saidie/train_cm1.slurm)
jc=$(sbatch --parsable --time=01:00:00 --dependency=afterok:$jb --job-name=F7_C --mem=48G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE $CA --init-from $R/$n --reuse-ae $R/$n $C" cluster/saidie/train_cm1.slurm)
je=$(sbatch --parsable --time=00:25:00 --dependency=afterany:$jc --export=ALL,OUT=v_eval_det_f4,RUNS="${n}_stagec F4_conv416_mixer512_pred_stagec F5_conv416_mixer512_stagec g_local48_stagec",EXTRA_EVAL="--ref-norm-run \$HOME/cm1_runs/um_local_stagec" cluster/saidie/vit_eval.slurm)
echo "F7: B $jb, C $jc, eval $je"
