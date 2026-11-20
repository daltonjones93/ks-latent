#!/bin/bash
# 9c (user: "try the pooled CRPS term on the stochastic Stage C for F6"): F6's Stage B output (F6_conv416_mixer512_gelu_pred_all: F6's autoencoder and
# propagator before Stage C) -> F6's Stage C (30k steps, k 45, batch 4, 4 CRPS samples, unfrozen autoencoder, lr 1e-4) PLUS the fair CRPS of the ensemble's
# neighbourhood-pooled |w| and condensate exceedances (2, 4, 8, 16 km; thresholds 2 m/s and 0.1 g/kg), weight 1 or 3 (the base CRPS is about 0.2 in these
# units; the pooled CRPS is of the order 0.02-0.07). The control is the existing F6 Stage C. Time: F6's Stage C took 90 min; +pooling about 15%.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/pool_crps.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C+$HOME/cm1_cache_C2+$HOME/cm1_cache_D"
R=$HOME/cm1_runs; SRC=$R/F6_conv416_mixer512_gelu_pred_all
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mmap --steps-b 40000 --mixer-variant gelu_ln"
CA="--ae-kind conv --hidden 416 --prop-body mixer --prop-hidden 512 --window-b 48"
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"
ALL=""
for w in 1 3; do
  n=F11_poolcrps_w$w
  jc=$(sbatch --parsable --time=02:05:00 --job-name=$n --mem=56G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE $CA --init-from $SRC --reuse-ae $SRC $C --c-pool-crps-weight $w" cluster/saidie/train_cm1.slurm)
  echo "$n: $jc"; ALL="$ALL:$jc"
done
je=$(sbatch --parsable --time=00:30:00 --dependency=afterany$ALL --export=ALL,OUT=v_eval_poolcrps,RUNS="F11_poolcrps_w1_stagec F11_poolcrps_w3_stagec F6_conv416_mixer512_gelu_pred_all_stagec F4_conv416_mixer512_pred_stagec g_local48_stagec",EXTRA_EVAL="--ref-norm-run ${HOME}/cm1_runs/um_local_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
