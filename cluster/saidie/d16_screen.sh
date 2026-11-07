#!/bin/bash
# D16: model screen on ens_B (see docs/cm1_plan.md). Five runs (S0, Stage C on S0, S1-S3) chained on ONE GPU (afterany), so the
# screen never holds more than one H200; round 2 (jobs 8753-8757) uses the other.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/d16_screen.sh [AFTER_JOBID]'
set -euo pipefail
cd "$HOME/latent_DA"
mkdir -p logs
P=ens_B_
VAL="${P}mem025+${P}mem053+${P}mem055+${P}mem058"
TEST="${P}mem003+${P}mem049+${P}mem016+${P}mem054+${P}mem022+${P}mem017+${P}mem050+${P}mem030+${P}mem011+${P}mem044"
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST"
ENV="--env-file $HOME/cm1_env/ens_B.json"
[ -f "$HOME/cm1_env/ens_B.json" ] || { echo "missing $HOME/cm1_env/ens_B.json"; exit 1; }
n=$(ls "$HOME"/cm1_cache_B/ens_B__mem*.npz | wc -l)
[ "$n" -eq 60 ] || { echo "only $n ens_B members cached"; exit 1; }
dep=${1:+--dependency=afterany:$1}
S0=$HOME/cm1_runs/d16_s0_base
STAGEC="--init-from $S0 --reuse-ae $S0 --stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2"   # as job 8753 on ens_A
for spec in "d16_s0_base|" "d16_s0_stagec|$STAGEC" "d16_s1_env|$ENV" "d16_s2_mirror|--mirror" "d16_s3_env_mirror|$ENV --mirror"; do
  run=${spec%%|*}; extra=${spec#*|}
  j=$(sbatch --parsable ${dep:-} --mem=160G \
      --export=ALL,DATA="cache:$HOME/cm1_cache_B",RUN="$run",EXTRA="$BASE $extra" cluster/saidie/train_cm1.slurm)
  echo "$run: job $j"
  dep="--dependency=afterany:$j"
done
