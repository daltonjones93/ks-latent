#!/bin/bash
# Speed of the CM1 training stages on the CPU (conv 2.6 M autoencoder + mixer 0.75 M, 121 storms): a few steps of Stage A, B and C.
set -uo pipefail
. ~/venvs/cm1/bin/activate
cd "$HOME/latent_DA"; export PYTHONPATH=$PWD OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
s0=$(date +%s)
python scripts/cm1/train_cm1.py --profile full --device cpu --threads ${OMP_NUM_THREADS} --source "$DATA" --out "$HOME/cm1_runs/cpu_bench" \
  --latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 2 \
  --ae-kind conv --hidden 256 --nested-frac 0.5 --steps-a 200 --prop-body mixer --prop-hidden 128 --window-b 48 --steps-b 60 \
  --stage-c-steps 20 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --stage-c-lr 1e-4 2>&1 | grep -E "\[A\] step|\[B\] step|\[C\] step|data on|loaded|done in|Error|error" | cut -c1-170
echo "TOTAL $(( $(date +%s) - s0 )) s"
