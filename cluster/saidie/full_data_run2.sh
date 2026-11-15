#!/bin/bash
# 8e: the best of the conv size x mixer size grid (cs_h416_m512: conv hidden 416 = 5.2 M, mixer width 512 ~ 12 M) on the same 325 training storms as F1-F3
# (so they compare directly), with (F4) and without (F5) the Stage A one-step prediction term (the clear gain of F1 over F2).
# Same recipe as F3: nested dropout 0.5, Stage A 60k steps (lr 1e-3, as fpc_nested), Stage B 40k, Stage C unfrozen 30k (lr 1e-4, batch 4), memory-mapped caches.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/full_data_run2.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C"
R=$HOME/cm1_runs
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mmap --steps-b 40000"
CONV="--ae-kind conv --hidden 416 --nested-frac 0.5 --steps-a 60000 --prop-body mixer --prop-hidden 512 --window-b 48"
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"
ALL=""
for spec in "F4_conv416_mixer512_pred|--w-pred-a 1 --pred-k-a 1" "F5_conv416_mixer512|"; do
  IFS='|' read -r n x <<< "$spec"
  jb=$(sbatch --parsable --time=02:30:00 --job-name=$n --mem=128G --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$BASE $CONV $x" cluster/saidie/train_cm1.slurm)
  jc=$(sbatch --parsable --time=02:15:00 --dependency=afterok:$jb --job-name=${n}_c --mem=128G --export=ALL,DATA="$DATA",RUN=${n}_stagec,EXTRA="$BASE --ae-kind conv --hidden 416 --prop-body mixer --prop-hidden 512 --init-from $R/$n --reuse-ae $R/$n $C" cluster/saidie/train_cm1.slurm)
  echo "$n: A+B $jb, C $jc"; ALL="$ALL:$jc"
done
je=$(sbatch --parsable --time=00:45:00 --dependency=afterany$ALL --export=ALL,OUT=v_eval_full2,RUNS="F4_conv416_mixer512_pred_stagec F5_conv416_mixer512_stagec F1_vit75_mlp_pred_stagec F3_conv_mixer_stagec um_local_stagec g_local48_stagec",EXTRA_EVAL="--ref-norm-run \$HOME/cm1_runs/um_local_stagec" cluster/saidie/vit_eval.slurm)
echo "eval: $je"
