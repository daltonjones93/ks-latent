#!/bin/bash
# 10d / 10a, 2026-11-23 afternoon: Stage A has become ~2.5-3x slower on the shared node (F16: 60k steps in 10 047 s; F6s4: A ~2.4 h), so the
# 3 h 15 A+B jobs ran out: F16 finished Stage A (ae.pt saved) and was killed before B; F6s4 was killed in B at 30k/40k; F16s1 (running) will
# likely miss as well; F17 (4 h 45, bigger mixer) would. Running jobs' limits cannot be raised (permission denied). New chains, one job
# per stage, reusing the saved autoencoders: B 2 h 30, C 3 h 30 (the EMA Stage C needed > 2 h 40), A-only 4 h 15.
# Cancels the chains that can no longer run (F16_stagec, F16s1_stagec, F6s4_stagec, F17 + C, the two evaluations, the 12 design jobs).
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/resubmit_new_storms2.sh'
set -euo pipefail
cd "$HOME/latent_DA"
scancel 9825 9827 9802 9828 9829 9830 9831 9806 9807 9808 9809 9810 9811 9812 9813 9814 9815 9816 9817 || true
B=ens_B_; A2=ens_A2_
VAL0="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST0="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST0="$TEST0+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST0="$TEST0+${B}mem$i"; done
NV=$(python3 -c "import json;print('+'.join(json.load(open('docs/cm1_split_E_final.json'))['val']))")
NT=$(python3 -c "import json;print('+'.join(json.load(open('docs/cm1_split_E_final.json'))['test']))")
L_=/tmp/c838596879_cm1; R=$HOME/cm1_runs; F6=$R/F6_conv416_mixer512_gelu_pred_all
EDIRS="cm1_cache_E1 cm1_cache_E_pilot4 cm1_cache_E2 cm1_cache_E3 cm1_cache_E4 cm1_cache_E5"
OLD="cache:$L_/cm1_cache+$L_/cm1_cache_A2+$L_/cm1_cache_B+$L_/cm1_cache_C+$L_/cm1_cache_C2+$L_/cm1_cache_D"
NEW="$OLD"; for d in $EDIRS; do NEW="$NEW+$L_/$d"; done
COM="--latent-channels 32 --k-max 45 --k-sampling uniform --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mmap --steps-b 40000 --mixer-variant gelu_ln"
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"
gpu() { local dep=${3:+--dependency=$3}; sbatch --parsable --time=$2 $dep --job-name=$1 --mem=64G --export=ALL,DATA="$5",RUN=$1,EXTRA="$4" cluster/saidie/train_cm1.slurm; }
A_() { echo "--ae-kind conv --hidden 416 --prop-body mixer --prop-hidden $1 --window-b 48 --nested-frac 0.5 --steps-a 60000 --w-pred-a 1 --pred-k-a 1 --seed $2"; }
# new-storm arms
NB="$COM --val-members $VAL0+$NV --test-members $TEST0+$NT --normalizer-from $R/F16"   # same data as F16: reuse its normalizer (no full re-read)
b16=$(gpu F16 02:30:00 "" "$NB $(A_ 512 0) --reuse-ae $R/F16" "$NEW")
c16=$(gpu F16_stagec 03:30:00 afterok:$b16 "$NB $(A_ 512 0) --init-from $R/F16 --reuse-ae $R/F16 $C" "$NEW")
b16b=$(gpu F16s1 02:30:00 afterany:9826 "$NB $(A_ 512 1) --reuse-ae $R/F16s1" "$NEW")
c16b=$(gpu F16s1_stagec 03:30:00 afterok:$b16b "$NB $(A_ 512 1) --init-from $R/F16s1 --reuse-ae $R/F16s1 $C" "$NEW")
a17=$(gpu F17 04:15:00 "" "$NB $(A_ 1024 0) --stage-a-only" "$NEW")
b17=$(gpu F17 03:00:00 afterok:$a17 "$NB $(A_ 1024 0) --reuse-ae $R/F17" "$NEW")
c17=$(gpu F17_stagec 03:30:00 afterok:$b17 "$NB $(A_ 1024 0) --init-from $R/F17 --reuse-ae $R/F17 $C" "$NEW")
# seed 4 of F6 (old storms)
OB="$COM --val-members $VAL0 --test-members $TEST0 --normalizer-from $F6"
b4=$(gpu F6s4 02:30:00 "" "$OB $(A_ 512 4) --reuse-ae $R/F6s4" "$OLD")
c4=$(gpu F6s4_stagec 03:30:00 afterok:$b4 "$OB $(A_ 512 4) --init-from $R/F6s4 --reuse-ae $R/F6s4 $C" "$OLD")
echo "F16 B $b16 C $c16; F16s1 B $b16b C $c16b; F17 A $a17 B $b17 C $c17; F6s4 B $b4 C $c4"
cpu() { sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=120G --time=$2 --job-name=$1 --output=logs/$1_%j.out --dependency=$3 \
  --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA; $4"; }
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"; for d in $EDIRS; do SRC="$SRC+$HOME/$d"; done
RUNS="$R/F6_conv416_mixer512_gelu_pred_all_stagec $R/F16_stagec $R/F16s1_stagec pool2=$R/F16_stagec+$R/F16s1_stagec $R/F17_stagec $R/g_local48_stagec"
for key in test old_test; do
  cpu eval_new_$key 02:30:00 afterany:$c16:$c16b:$c17 "python scripts/cm1/twoscale_eval.py --runs $RUNS --source $SRC --members-json docs/cm1_split_E_final.json \
--members-key $key --n-test 99 --lags 0,5,15,30,60 --samples 16 --threads 16 --ref-norm-run $R/um_local_stagec --out \$HOME/cm1_da/eval_new_$key"
done
M="${F6}_stagec $R/F6s1_stagec $R/F6s2_stagec $R/F6s3_stagec $R/F6s4_stagec $R/F6s5_stagec"
D=""
for T in 0.7 1.0 1.3; do for I in 0.0 0.1; do for SP in val test; do
  [ $SP = val ] && TL=01:15:00 || TL=03:30:00
  D="$D $(cpu design_$SP $TL afterok:$c4 "python scripts/cm1/ensemble_design.py --models $M --members 8 --temp $T --ic-noise $I --split $SP --out \$HOME/cm1_da/design/${SP}_T${T}_I${I}.json")"
done; done; done
echo "design$D"
