#!/bin/bash
# FPc: the column model with a readable latent law (docs/cm1_plan.md, track FP, FPc). ONE GPU, sequential:
#   1. fpc_law_reuse      full_v1's autoencoder, law propagator (L1 1e-5, residual weight 0.1), Stage B as full_v1
#      -> fpc_law_reuse_stagec (Stage C 15 000 steps, k 45) -> law_report
#   2. fpc_law_l1e4       the same with L1 1e-4 (sparser) -> Stage C -> law_report
#   3. fpc_nested         new autoencoder with nested dropout (importance-ordered channels, 50% of samples
#      truncated) + law (L1 1e-5) -> Stage C with nested dropout -> law_report
#   ssh saidie 'cd ~/latent_DA && AFTER=<job> bash -l cluster/saidie/fpc_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST"
LAW="--prop-body law --law-res-weight 0.1 --law-cap 5"   # cap: 8999 blew up without it (see the plan)
C="--stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2"
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
if [ -z "${GLOBAL_ONLY:-}" ]; then
sub fpc_law_reuse "--reuse-ae $R/full_v1 $LAW --law-l1 1e-5" afterany
sub fpc_law_reuse_stagec "--init-from $R/fpc_law_reuse --reuse-ae $R/fpc_law_reuse $C"
report fpc_law_reuse_stagec
sub fpc_law_l1e4 "--reuse-ae $R/full_v1 $LAW --law-l1 1e-4" afterany
sub fpc_law_l1e4_stagec "--init-from $R/fpc_law_l1e4 --reuse-ae $R/fpc_law_l1e4 $C"
report fpc_law_l1e4_stagec
sub fpc_nested "$LAW --law-l1 1e-5 --nested-frac 0.5" afterany
sub fpc_nested_stagec "--init-from $R/fpc_nested --reuse-ae $R/fpc_nested $C --nested-frac 0.5"
report fpc_nested_stagec
fi
# 4. hybrid local + global: + G <z> (channel means over the domain). Trained on whole-domain windows (200 sites)
#    so the mean in training is the domain mean used in evaluation; with a whole-domain local control (no G), so the
#    comparison is not confounded by the window width. GLOBAL_ONLY=1 submits only these.
WD="--window-b 200 --stage-c-batch 6"
sub fpc_law_full "--reuse-ae $R/full_v1 $LAW --law-l1 1e-5 --window-b 200" afterany
sub fpc_law_full_stagec "--init-from $R/fpc_law_full --reuse-ae $R/fpc_law_full $C $WD"
report fpc_law_full_stagec
sub fpc_law_global "--reuse-ae $R/full_v1 $LAW --law-l1 1e-5 --law-global --window-b 200" afterany
sub fpc_law_global_stagec "--init-from $R/fpc_law_global --reuse-ae $R/fpc_law_global $C $WD"
report fpc_law_global_stagec
