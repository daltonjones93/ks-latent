#!/bin/bash
# 2026-11-25 (evening): F16s1 B (9917) and F17 B (9927) hung for 3.5 h / 5 h in encode_members right after reusing their
# autoencoders. Cause (diag_encode_hang.sh, 9946/9947; same at 64G and 480G): reading the memory-mapped caches page by page ran
# ~10x slower than os.pread on the loaded node, almost all of it kernel time; ks_latent/cm1/data.py now reads with pread.
# The new-storm evaluations A (9921/9924) crashed on a CPU/GPU kernel mismatch in twoscale_eval.lowpass (fixed 5234aa9).
# Cancels the dead dependents and resubmits: F16s1 B -> C, F17 B -> C, evaluations A now, B after F16s1 C, C after F17 C.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/resubmit_new_storms4.sh'
set -euo pipefail
cd "$HOME/latent_DA"
scancel 9918 9922 9925 9928 9929 9930 9946 9947 || true
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
b1=$(gpu F16s1 03:30:00 "" "$NB $(A_ 512 1) --reuse-ae $R/F16s1")
c1=$(gpu F16s1_stagec 03:30:00 afterok:$b1 "$NB $(A_ 512 1) --init-from $R/F16s1 --reuse-ae $R/F16s1 $C")
b17=$(gpu F17 05:00:00 "" "$NB $(A_ 1024 0) --reuse-ae $R/F17")
c17=$(gpu F17_stagec 04:30:00 afterok:$b17 "$NB $(A_ 1024 0) --init-from $R/F17 --reuse-ae $R/F17 $C")
echo "F16s1 B $b1 C $c1; F17 B $b17 C $c17"
ev() { sbatch --parsable --account=chen-saidie --partition=general --gres=gpu:h200:1 --cpus-per-task=16 --mem=120G --time=04:00:00 \
  --job-name=$1 --output=logs/$1_%j.out ${2:+--dependency=$2} \
  --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 ; cd \$HOME/latent_DA; $3"; }
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"; for d in $EDIRS; do SRC="$SRC+$HOME/$d"; done
EV="--source $SRC --members-json docs/cm1_split_E_final.json --n-test 99 --lags 0,5,15,30,60 --samples 16 --threads 16 --ref-norm-run $R/um_local_stagec --device cuda"
for key in test old_test; do
  p=ev_$key; [ $key = old_test ] && p=ev_old_test
  ev ${p}_A "" "python scripts/cm1/twoscale_eval.py --runs ${F6}_stagec $R/F16_stagec $R/g_local48_stagec --members-key $key $EV --out \$HOME/cm1_da/eval_new_$key/A"
  ev ${p}_B afterok:$c1 "python scripts/cm1/twoscale_eval.py --runs $R/F16s1_stagec pool2=$R/F16_stagec+$R/F16s1_stagec --members-key $key $EV --out \$HOME/cm1_da/eval_new_$key/B"
  ev ${p}_C afterok:$c17 "python scripts/cm1/twoscale_eval.py --runs $R/F17_stagec --members-key $key $EV --out \$HOME/cm1_da/eval_new_$key/C"
done
