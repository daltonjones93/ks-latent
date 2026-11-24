#!/bin/bash
# 2026-11-24: what resubmit_new_storms2 left. F16 done (B + C). F16s1's Stage A was killed at 57k/60k (no ae.pt) -> A-only 4 h 15, B 2 h 30,
# C 3 h 30. F17's Stage B (mixer 1024) was killed at 38k/40k after 3 h -> B 4 h 30, C 4 h 30. The two new-storm evaluations waited on
# chains that can no longer run -> cancelled; an evaluation of F16 now (F6, F16, g_local48) and the full one after the new chains.
# design_val (6 models) needed > 1 h 15 (12 of 18 cases in 69 min) -> rerun with 2 h 15.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/resubmit_new_storms3.sh'
set -euo pipefail
cd "$HOME/latent_DA"
scancel 9862 9865 9868 9869 || true
B=ens_B_; A2=ens_A2_
VAL0="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST0="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST0="$TEST0+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST0="$TEST0+${B}mem$i"; done
NV=$(python3 -c "import json;print('+'.join(json.load(open('docs/cm1_split_E_final.json'))['val']))")
NT=$(python3 -c "import json;print('+'.join(json.load(open('docs/cm1_split_E_final.json'))['test']))")
L_=/tmp/c838596879_cm1; R=$HOME/cm1_runs; F6=$R/F6_conv416_mixer512_gelu_pred_all
EDIRS="cm1_cache_E1 cm1_cache_E_pilot4 cm1_cache_E2 cm1_cache_E3 cm1_cache_E4 cm1_cache_E5"
NEW="cache:$L_/cm1_cache+$L_/cm1_cache_A2+$L_/cm1_cache_B+$L_/cm1_cache_C+$L_/cm1_cache_C2+$L_/cm1_cache_D"; for d in $EDIRS; do NEW="$NEW+$L_/$d"; done
COM="--latent-channels 32 --k-max 45 --k-sampling uniform --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mmap --steps-b 40000 --mixer-variant gelu_ln"
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"
gpu() { local dep=${3:+--dependency=$3}; sbatch --parsable --time=$2 $dep --job-name=$1 --mem=64G --export=ALL,DATA="$NEW",RUN=$1,EXTRA="$4" cluster/saidie/train_cm1.slurm; }
A_() { echo "--ae-kind conv --hidden 416 --prop-body mixer --prop-hidden $1 --window-b 48 --nested-frac 0.5 --steps-a 60000 --w-pred-a 1 --pred-k-a 1 --seed $2"; }
NB="$COM --val-members $VAL0+$NV --test-members $TEST0+$NT --normalizer-from $R/F16"
a1=$(gpu F16s1 04:15:00 "" "$NB $(A_ 512 1) --stage-a-only")
b1=$(gpu F16s1 02:30:00 afterok:$a1 "$NB $(A_ 512 1) --reuse-ae $R/F16s1")
c1=$(gpu F16s1_stagec 03:30:00 afterok:$b1 "$NB $(A_ 512 1) --init-from $R/F16s1 --reuse-ae $R/F16s1 $C")
b17=$(gpu F17 04:30:00 "" "$NB $(A_ 1024 0) --reuse-ae $R/F17")
c17=$(gpu F17_stagec 04:30:00 afterok:$b17 "$NB $(A_ 1024 0) --init-from $R/F17 --reuse-ae $R/F17 $C")
echo "F16s1 A $a1 B $b1 C $c1; F17 B $b17 C $c17"
cpu() { sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=120G --time=$2 --job-name=$1 --output=logs/$1_%j.out ${3:+--dependency=$3} \
  --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA; $4"; }
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"; for d in $EDIRS; do SRC="$SRC+$HOME/$d"; done
EV="--source $SRC --members-json docs/cm1_split_E_final.json --n-test 99 --lags 0,5,15,30,60 --samples 16 --threads 16 --ref-norm-run $R/um_local_stagec"
for key in test old_test; do
  cpu f16_$key 02:30:00 "" "python scripts/cm1/twoscale_eval.py --runs ${F6}_stagec $R/F16_stagec $R/g_local48_stagec --members-key $key $EV --out \$HOME/cm1_da/f16_$key"
  cpu eval_new_$key 03:00:00 afterany:$c1:$c17 "python scripts/cm1/twoscale_eval.py --runs ${F6}_stagec $R/F16_stagec $R/F16s1_stagec pool2=$R/F16_stagec+$R/F16s1_stagec $R/F17_stagec $R/g_local48_stagec --members-key $key $EV --out \$HOME/cm1_da/eval_new_$key"
done
M="${F6}_stagec $R/F6s1_stagec $R/F6s2_stagec $R/F6s3_stagec $R/F6s4_stagec $R/F6s5_stagec"
for T in 0.7 1.0 1.3; do for I in 0.0 0.1; do
  cpu design_val 02:15:00 "" "python scripts/cm1/ensemble_design.py --models $M --members 8 --temp $T --ic-noise $I --split val --out \$HOME/cm1_da/design/val_T${T}_I${I}.json"
done; done
