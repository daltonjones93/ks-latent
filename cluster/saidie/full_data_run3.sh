#!/bin/bash
# 8f: F6 = the GELU + LayerNorm twin of F4 (conv 5.2 M + mixer ~12 M + Stage A one-step prediction term) on ALL the new storms:
# ens_A + ens_A2 + ens_B (121) + ens_C (360: ~/cm1_cache_C and _C2) + ens_D (120): 601 storms, 573 for training; the same 6 validation and
# 22 test storms. The ablation (8b) found GELU + LayerNorm equal or about 2% better than SwiGLU + RMSNorm on the large pair. Same steps as F4
# (Stage A 60k, Stage B 40k, Stage C 30k, unfrozen), so the comparison with F4 is the data (325 to 573 storms) and the block details.
# Time limits from F4 (A 52 min + B 75 min + encoding; C 86 min) with margin.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/full_data_run3.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C+$HOME/cm1_cache_C2+$HOME/cm1_cache_D"
R=$HOME/cm1_runs; n=F6_conv416_mixer512_gelu_pred_all
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mmap --steps-b 40000 --mixer-variant gelu_ln"
CA="--ae-kind conv --hidden 416 --prop-body mixer --prop-hidden 512 --window-b 48"
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"
jb=$(sbatch --parsable --time=02:45:00 --job-name=F6_AB --mem=48G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE $CA --nested-frac 0.5 --steps-a 60000 --w-pred-a 1 --pred-k-a 1" cluster/saidie/train_cm1.slurm)
jc=$(sbatch --parsable --time=01:50:00 --dependency=afterok:$jb --job-name=F6_C --mem=48G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE $CA --init-from $R/$n --reuse-ae $R/$n $C" cluster/saidie/train_cm1.slurm)
je=$(sbatch --parsable --time=00:25:00 --dependency=afterany:$jc --export=ALL,OUT=v_eval_f6,RUNS="${n}_stagec F4_conv416_mixer512_pred_stagec F5_conv416_mixer512_stagec F1_vit75_mlp_pred_stagec F3_conv_mixer_stagec um_local_stagec g_local48_stagec",EXTRA_EVAL="--ref-norm-run ${HOME}/cm1_runs/um_local_stagec" cluster/saidie/vit_eval.slurm)
echo "F6: A+B $jb, C $jc, eval $je"
