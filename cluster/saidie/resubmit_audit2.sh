#!/bin/bash
# 10b, 2026-11-23: what the first node-local chain left over. Done: seeds 0, 2, 3, 5 (F6s2/3/5 with results). F6s1_stagec (Stage C 2 h 10 limit) and
# F15_k90 (3 h 30) finished all 30000 Stage C steps and saved ae.pt / prop.pt, then timed out in the final scoring (no results.json) ->
# `--eval-only` jobs write it. Seed 4, EMA (blocked behind seed 1's dependency) and the evaluations / design jobs are resubmitted;
# the Stage C limit goes to 2 h 40.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/resubmit_audit2.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
L_=/tmp/c838596879_cm1
DATA="cache:$L_/cm1_cache+$L_/cm1_cache_A2+$L_/cm1_cache_B+$L_/cm1_cache_C+$L_/cm1_cache_C2+$L_/cm1_cache_D"
R=$HOME/cm1_runs; F6=$R/F6_conv416_mixer512_gelu_pred_all
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mmap --steps-b 40000 --mixer-variant gelu_ln --normalizer-from $F6"
CA="--ae-kind conv --hidden 416 --prop-body mixer --prop-hidden 512 --window-b 48"
C="--stage-c-steps 30000 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"
gpu() { local dep=${3:+--dependency=afterok:$3}; sbatch --parsable --time=$2 $dep --job-name=$1 --mem=56G --export=ALL,DATA="$DATA",RUN=$1,EXTRA="$4" cluster/saidie/train_cm1.slurm; }
e1=$(gpu F6s1_stagec 00:50:00 "" "$BASE $CA --eval-only")
e2=$(gpu F15_k90_stagec 00:50:00 "" "$BASE $CA --eval-only")
ab=$(gpu F6s4 03:15:00 "" "$BASE $CA --nested-frac 0.5 --steps-a 60000 --w-pred-a 1 --pred-k-a 1 --seed 4")
c4=$(gpu F6s4_stagec 02:40:00 $ab "$BASE $CA --init-from $R/F6s4 --reuse-ae $R/F6s4 $C --stage-c-k 45 --seed 4")
em=$(gpu F14_ema_stagec 02:40:00 "" "$BASE $CA --init-from $F6 --reuse-ae $F6 $C --stage-c-k 45 --c-ema 0.999")
echo "eval-only s1 $e1, k90 $e2; seed 4 A+B $ab C $c4; EMA $em"
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
EV="--source $SRC --n-test 22 --threads 16 --ref-norm-run $R/um_local_stagec"
cpu() { sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=120G --time=$2 --job-name=$1 --output=logs/$1_%j.out --dependency=$3 \
  --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA; $4"; }
L="--lags 0,5,15,30,60,90 --samples 24"
f1=$(cpu audit_eval1 01:45:00 afterok:$e1 "python scripts/cm1/twoscale_eval.py --runs ${F6}_stagec $R/F6s1_stagec $R/F6s2_stagec $R/F6s3_stagec pool3=${F6}_stagec+$R/F6s1_stagec+$R/F6s2_stagec $EV $L --out \$HOME/cm1_da/audit_seeds")
f2=$(cpu audit_eval2 01:30:00 afterok:$e2:$em "python scripts/cm1/twoscale_eval.py --runs ${F6}_stagec $R/F14_ema_stagec $R/F15_k90_stagec $R/F7_conv416_mixer512_det_stagec $EV $L --out \$HOME/cm1_da/audit_arms")
M="${F6}_stagec $R/F6s1_stagec $R/F6s2_stagec $R/F6s3_stagec $R/F6s4_stagec $R/F6s5_stagec"
D=""
for T in 0.7 1.0 1.3; do for I in 0.0 0.1; do for SP in val test; do
  [ $SP = val ] && TL=01:15:00 || TL=03:30:00
  D="$D $(cpu design_$SP $TL afterok:$e1:$c4 "python scripts/cm1/ensemble_design.py --models $M --members 8 --temp $T --ic-noise $I --split $SP --out \$HOME/cm1_da/design/${SP}_T${T}_I${I}.json")"
done; done; done
echo "evals $f1 $f2, design$D"
