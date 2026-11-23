#!/bin/bash
# 10a, 2026-11-23 (third round): F14_ema_stagec timed out at Stage C step ~28k of 30k (2 h 40 limit; Stage C runs slower while the node is
# shared) -> rerun with 3 h 30. audit_eval1 (seeds/pool) timed out and is superseded by pool_now (10b-result); audit_eval2 never ran.
# Q4 (k90) and Q3 (EMA) are scored alone, 24 members, 66 cases, lags to 90 min, as pool_now (F6's own scores are in pool_now_a).
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/resubmit_audit3.sh'
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
em=$(sbatch --parsable --time=03:30:00 --job-name=F14_ema_stagec --mem=56G --export=ALL,DATA="$DATA",RUN=F14_ema_stagec,EXTRA="$BASE $CA --init-from $F6 --reuse-ae $F6 $C --stage-c-k 45 --c-ema 0.999" cluster/saidie/train_cm1.slurm)
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
EV="--source $SRC --n-test 22 --threads 16 --ref-norm-run $R/um_local_stagec --lags 0,5,15,30,60,90 --samples 24"
cpu() { sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=120G --time=01:30:00 --job-name=$1 --output=logs/$1_%j.out $2 \
  --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA; \
python scripts/cm1/twoscale_eval.py --runs $3 $EV --out \$HOME/cm1_da/$1"; }
k=$(cpu audit_k90 "" "$R/F15_k90_stagec")
e=$(cpu audit_ema "--dependency=afterok:$em" "$R/F14_ema_stagec")
echo "EMA $em; evals k90 $k, EMA $e"
