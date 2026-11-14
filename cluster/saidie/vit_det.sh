#!/bin/bash
# 8a: the dense-MLP propagator on the ViT latent WITHOUT noise channels (user: why noise channels? train it deterministically).
# Same autoencoder (v_win: ViT 198 M, windowed), same body (mlp, width 256, 1.65 M minus the 4 noise inputs), but Stage B with the
# MSE rollout loss in latent space (noise_channels 0, no CRPS) and Stage C with the decoded MSE rollout loss, autoencoder
# unfrozen exactly as u_win_mlp (lr 5e-5, batch 4, nested dropout 0.5). Compare with u_win_mlp on RMSE of the (sample-mean) forecast,
# CRPS (MAE for a single deterministic forecast), spectra and spread.   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/vit_det.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
R=$HOME/cm1_runs
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --tf32"
A="--ae-kind vit --nested-frac 0.5 --vit-radius 2 --prop-body mlp --window-a 64"
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 5e-5"
jb=$(sbatch --parsable --time=00:45:00 --job-name=d_win_mlp --mem=200G --export=ALL,DATA="$DATA",RUN=d_win_mlp,EXTRA="$BASE $A --reuse-ae $R/v_win" cluster/saidie/train_cm1.slurm)
jc=$(sbatch --parsable --time=01:00:00 --dependency=afterok:$jb --job-name=d_win_mlp_c --mem=200G --export=ALL,DATA="$DATA",RUN=d_win_mlp_stagec,EXTRA="$BASE --init-from $R/d_win_mlp --reuse-ae $R/d_win_mlp $C" cluster/saidie/train_cm1.slurm)
je=$(sbatch --parsable --time=00:30:00 --dependency=afterany:$jc --export=ALL,OUT=v_eval_det,RUNS="d_win_mlp_stagec u_win_mlp_stagec" cluster/saidie/vit_eval.slurm)
echo "d_win_mlp: B $jb, C $jc, eval $je"
