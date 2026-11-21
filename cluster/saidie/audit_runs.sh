#!/bin/bash
# 10a (user, 2026-11-21: audit the CM1 work, look for missed promising directions for FORECAST and MODEL ACCURACY, plan and queue them).
# Four questions, each untested on the F-series (conv 416 + GELU mixer 512, Stage A prediction term, 573 storms):
#   Q1 ensemble size at forecast time: F6 has only ever been scored with 8 members. Fair CRPS does not depend on the size, but the
#      ensemble-mean error does (a calibrated M-member mean has error variance sigma^2 (1 + 1/M): 8 -> 32 members should cut its RMSE ~4%,
#      more than F7's deterministic advantage of 2-3%), and the vote gets finer. No training. (CPU jobs.)
#   Q2 seed variance and multi-model ensembles: every F-series comparison (F6 vs F4 1.6%, 9b arms <= 1%, 9c 0.5-2.5%) is ONE seed; the
#      only seed spread measured (conv-128, 2026-11-06) was ~1%. Two more F6 seeds give the spread, and pooling 3 x 8 members tests
#      the D14a result (averaging independently trained propagators was the best single idea then, 5-10%), never applied to F-series.
#   Q3 weight EMA in Stage C (decay 0.999; new, tested): a standard, cheap accuracy lever, never tried here. From F6's Stage B.
#   Q4 Stage C horizon 90 min (F6 is trained to 45 and judged at 60; SM4c's k90 was the only arm that helped there). From F6's Stage B.
# Final evaluation on ALL 22 test storms x 3 starts (66 cases, not the 10-storm 30-case set used since 9b), lags to 90 min, 24 members.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/audit_runs.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C+$HOME/cm1_cache_C2+$HOME/cm1_cache_D"
R=$HOME/cm1_runs; F6=F6_conv416_mixer512_gelu_pred_all; S6=$R/$F6
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mmap --steps-b 40000 --mixer-variant gelu_ln"
CA="--ae-kind conv --hidden 416 --prop-body mixer --prop-hidden 512 --window-b 48"
C="--stage-c-steps 30000 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
EV="--source $SRC --n-test 22 --threads 16 --ref-norm-run $R/um_local_stagec"
cpu() { # name time command
  sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=96G --time=$2 --job-name=$1 --output=logs/$1_%j.out ${DEP:+--dependency=$DEP} \
    --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA; $3"; }
gpu() { # name time dep extra
  local dep=$3; sbatch --parsable --time=$2 ${dep:+--dependency=afterok:$dep} --job-name=$1 --mem=56G --export=ALL,DATA="$DATA",RUN=$1,EXTRA="$4" cluster/saidie/train_cm1.slurm; }

# Q1: ensemble size, no training (66 cases; F6 at 8 / 16 / 32 members with the references at 8; 64 members on its own)
q1a=$(cpu ens_size 01:10:00 "for M in 8 16 32; do RUNS=${S6}_stagec; [ \$M = 8 ] && RUNS=\"${S6}_stagec $R/F7_conv416_mixer512_det_stagec $R/F4_conv416_mixer512_pred_stagec $R/g_local48_stagec\"; python scripts/cm1/twoscale_eval.py --runs \$RUNS $EV --lags 0,5,15,30,60,90 --samples \$M --out \$HOME/cm1_da/ens_size_\$M; done")
q1b=$(cpu ens_size64 01:00:00 "python scripts/cm1/twoscale_eval.py --runs ${S6}_stagec $EV --lags 5,15,30,60,90 --samples 64 --out \$HOME/cm1_da/ens_size_64")
echo "Q1: $q1a $q1b"

# Q2: two more F6 seeds (F6's own times: A+B 2 h 16, C 1 h 30; limits ~1.3x)
S=""
for sd in 1 2; do
  n=F6s${sd}; ab=$(gpu $n 02:55:00 "" "$BASE $CA --nested-frac 0.5 --steps-a 60000 --w-pred-a 1 --pred-k-a 1 --seed $sd")
  c=$(gpu ${n}_stagec 01:55:00 $ab "$BASE $CA --init-from $R/$n --reuse-ae $R/$n $C --stage-c-k 45 --seed $sd")
  echo "Q2 $n: A+B $ab, C $c"; S="$S:$c"
done
# Q3: EMA; Q4: horizon 90 (both from F6's Stage B, seed 0 as F6)
e=$(gpu F14_ema_stagec 01:55:00 "" "$BASE $CA --init-from $S6 --reuse-ae $S6 $C --stage-c-k 45 --c-ema 0.999"); echo "Q3: $e"
k=$(gpu F15_k90_stagec 03:20:00 "" "$BASE $CA --init-from $S6 --reuse-ae $S6 $C --stage-c-k 90"); echo "Q4: $k"

# final evaluation: 66 cases, 24 members (pool = 3 seeds x 8), lags to 90; two CPU jobs in parallel
L="--lags 0,5,15,30,60,90 --samples 24"
DEP="afterany$S:$e:$k" f1=$(cpu audit_eval1 01:20:00 "python scripts/cm1/twoscale_eval.py --runs ${S6}_stagec $R/F6s1_stagec $R/F6s2_stagec pool3=${S6}_stagec+$R/F6s1_stagec+$R/F6s2_stagec $EV $L --out \$HOME/cm1_da/audit_seeds")
DEP="afterany$S:$e:$k" f2=$(cpu audit_eval2 01:10:00 "python scripts/cm1/twoscale_eval.py --runs ${S6}_stagec $R/F14_ema_stagec $R/F15_k90_stagec $R/F7_conv416_mixer512_det_stagec $EV $L --out \$HOME/cm1_da/audit_arms")
echo "evals: $f1 $f2"
