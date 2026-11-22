#!/bin/bash
# 10d: the larger training on old + new (varied) storms (user, 2026-11-22: "larger trainings on more varied storms"; "make sure
# everything that should be running is running, and queue everything else"). Run once all E waves are screened and on Saidie.
# Arms (F6's recipe: conv 416 + GELU mixer, Stage A prediction term, A 60k / B 40k / C 30k):
#   F16    old 573 + new ~280 training storms, seed 0          F16s1  the same, seed 1 (seed spread; pool of 2)
#   F17    the same data, mixer width 1024 (47 M; F10's size: capacity may matter again on broader data), seed 0
# Validation / test: the old 6 / 22 storms + the new 18 / 36 (docs/cm1_split_E.json, minus storms flagged for domain-wide initiation).
# Evaluations (CPU): new test storms and old test storms, F6 / F16 / F16s1 / pool(F16, F16s1) / F17 / g_local48.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/train_new_storms.sh'
set -euo pipefail
cd "$HOME/latent_DA"
EDIRS="cm1_cache_E1 cm1_cache_E_pilot4 cm1_cache_E2 cm1_cache_E3 cm1_cache_E4 cm1_cache_E5"
for d in $EDIRS; do [ -d $HOME/$d ] || { echo "missing $HOME/$d"; exit 1; }; done
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
python3 - "$VAL" "$TEST" <<'PY'
import json, sys
from pathlib import Path
H = Path.home()
flag = set(x.strip() for x in open(H / "flagged_E.txt") if x.strip())
s = json.load(open("docs/cm1_split_E.json"))
have = {f.stem.replace("__", "_") for d in ("cm1_cache_E1", "cm1_cache_E_pilot4", "cm1_cache_E2", "cm1_cache_E3", "cm1_cache_E4", "cm1_cache_E5")
        for f in (H / d).glob("*.npz")}
new = {k: [n for n in s[k] if n not in flag and n in have] for k in ("val", "test")}
json.dump({"test": new["test"], "val": new["val"], "old_test": sys.argv[2].split("+")}, open("docs/cm1_split_E_final.json", "w"), indent=1)
open("/tmp/c838596879_newval", "w").write("+".join(new["val"])); open("/tmp/c838596879_newtest", "w").write("+".join(new["test"]))
print(f"new val {len(new['val'])}, new test {len(new['test'])}; flagged {len(flag)}")
PY
VAL="$VAL+$(cat /tmp/c838596879_newval)"; TEST="$TEST+$(cat /tmp/c838596879_newtest)"
L_=/tmp/c838596879_cm1
DATA="cache:$L_/cm1_cache+$L_/cm1_cache_A2+$L_/cm1_cache_B+$L_/cm1_cache_C+$L_/cm1_cache_C2+$L_/cm1_cache_D"
for d in $EDIRS; do DATA="$DATA+$L_/$d"; done
R=$HOME/cm1_runs
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mmap --steps-b 40000 --mixer-variant gelu_ln"
C="--stage-c-steps 30000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"
st=$(DIRS="$EDIRS" sbatch --parsable --export=ALL,DIRS="$EDIRS" --account=chen-saidie --partition=general --cpus-per-task=4 --mem=8G --time=01:30:00 --job-name=stage_E --output=logs/stage_E_%j.out cluster/saidie/stage_local.sh)
gpu() { sbatch --parsable --time=$2 --dependency=afterok:$3 --job-name=$1 --mem=64G --export=ALL,DATA="$DATA",RUN=$1,EXTRA="$4" cluster/saidie/train_cm1.slurm; }
arm() { # name seed mixer-width A+B-limit C-limit -> echoes the Stage C job
  local ab; ab=$(gpu $1 $4 $st "$BASE --ae-kind conv --hidden 416 --prop-body mixer --prop-hidden $3 --window-b 48 --nested-frac 0.5 --steps-a 60000 --w-pred-a 1 --pred-k-a 1 --seed $2")
  gpu ${1}_stagec $5 $ab "$BASE --ae-kind conv --hidden 416 --prop-body mixer --prop-hidden $3 --window-b 48 --init-from $R/$1 --reuse-ae $R/$1 $C --seed $2"; }
c16=$(arm F16 0 512 03:15:00 02:10:00); c16b=$(arm F16s1 1 512 03:15:00 02:10:00); c17=$(arm F17 0 1024 04:45:00 02:15:00)
echo "staging $st; F16 C $c16, F16s1 C $c16b, F17 C $c17"
SRC="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"; for d in $EDIRS; do SRC="$SRC+$HOME/$d"; done
RUNS="$R/F6_conv416_mixer512_gelu_pred_all_stagec $R/F16_stagec $R/F16s1_stagec pool2=$R/F16_stagec+$R/F16s1_stagec $R/F17_stagec $R/g_local48_stagec"
for key in test old_test; do
  sbatch --parsable --account=chen-saidie --partition=general --cpus-per-task=16 --mem=120G --time=02:30:00 --job-name=eval_new_$key \
    --output=logs/eval_new_${key}_%j.out --dependency=afterany:$c16:$c16b:$c17 \
    --wrap=". \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=; cd \$HOME/latent_DA; \
python scripts/cm1/twoscale_eval.py --runs $RUNS --source $SRC --members-json docs/cm1_split_E_final.json --members-key $key --n-test 99 \
--lags 0,5,15,30,60 --samples 16 --threads 16 --ref-norm-run $R/um_local_stagec --out \$HOME/cm1_da/eval_new_$key"
done
