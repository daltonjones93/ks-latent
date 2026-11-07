#!/bin/bash
# FPd: shape the column latent so the explicit law closes (docs/cm1_plan.md, track FP, FPd). ONE GPU, sequential,
# after FPc (needs fpc_law_reuse's Stage-B output). Stage C 20 000 steps, k 45, with: L1 1e-4; the residual-share
# weight ramped 0.1 -> 5 over the first half (the closed form must carry the tendency); sequential thresholding
# every 2500 steps from 40% (terms below 2% of the largest coupling removed, the autoencoder adapts).
#   1. fpd_reuse     from fpc_law_reuse (full_v1's autoencoder + law after Stage B)
#   2. fpd_smooth    1 + smoothness of the latent over sites (0.02; physical space, not the channel index)
#   3. fpd_scratch   a new autoencoder (Stage A) + law (Stage B), then Stage C as 2
#   ssh saidie 'cd ~/latent_DA && AFTER=<job> bash -l cluster/saidie/fpd_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST"
LAW="--prop-body law --law-res-weight 0.1 --law-cap 5"
FPD="--stage-c-steps 20000 --stage-c-k 45 --w-sfc-th 2 --c-law-l1 1e-4 --c-law-res-end 5 --c-law-prune-every 2500 --c-law-prune-rel 0.02"
R=$HOME/cm1_runs
prev=${AFTER:-}
sub() {   # name, extra args, dependency type
  local dep=${prev:+--dependency=${3:-afterok}:$prev}
  prev=$(sbatch --parsable ${dep:-} --mem=200G --export=ALL,DATA="$DATA",RUN=$1,EXTRA="$BASE $2" cluster/saidie/train_cm1.slurm)
  echo "$1: $prev"
}
report() {
  local dep=--dependency=afterok:$prev
  prev=$(sbatch --parsable $dep --account=chen-saidie --partition=general --gres=gpu:h200:1 --cpus-per-task=16 --mem=200G \
    --time=03:00:00 --job-name=cm1lawrep --output=logs/cm1lawrep_%j.out --wrap "set -e
[ -n \"\${CUDA_VISIBLE_DEVICES:-}\" ] || { echo 'CUDA_VISIBLE_DEVICES not set'; exit 1; }
. \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16
python -m pytest -q tests/unit/test_cm1_law.py -p no:cacheprovider
python scripts/cm1/law_report.py --run $R/$1 --source $DATA --device cuda")
  echo "report $1: $prev"
}
sub fpd_reuse "--init-from $R/fpc_law_reuse --reuse-ae $R/fpc_law_reuse $FPD" afterany
report fpd_reuse
sub fpd_smooth "--init-from $R/fpc_law_reuse --reuse-ae $R/fpc_law_reuse $FPD --c-site-smooth 0.02" afterany
report fpd_smooth
sub fpd_scratch_ab "$LAW --law-l1 1e-5" afterany
sub fpd_scratch "--init-from $R/fpd_scratch_ab --reuse-ae $R/fpd_scratch_ab $FPD --c-site-smooth 0.02"
report fpd_scratch
