#!/bin/bash
# DA evaluation of a trained run under the G0 protocol, chained on ONE GPU:
#   reflectivity head -> validation tuning of the latent model noise (one short study per value, --tuning)
#   -> test-storm study at the value with the lowest validation prior nMSE (da_tuned.slurm) -> paired
#   comparison with the headline. Outputs ~/cm1_da/tune_<LABEL>_noise_<n>, tune_<LABEL>_choice.json, da_<LABEL>.
#   ssh saidie 'cd ~/latent_DA && AFTER=<job> bash -l cluster/saidie/da_eval_run.sh RUN LABEL "0.05 0.1 0.15"'
set -euo pipefail
cd "$HOME/latent_DA"
RUN=${1:?run} LABEL=${2:?label} NOISES=${3:-"0.02 0.05 0.1 0.15 0.2 0.3"}
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
TUNE="--tuning --modes latent --n-members 32 --random-members --seeds 0 1 2 --obs-set th+refl+vr --boundary persistence"
dep=${AFTER:+--dependency=afterok:$AFTER}
r=$(sbatch --parsable ${dep:-} --time=01:00:00 --export=ALL,RUN=$RUN,DATA="$DATA" cluster/saidie/refl_head.slurm)
echo "$RUN reflectivity head: $r"
j=$r
for n in $NOISES; do
  j=$(sbatch --parsable --dependency=afterok:$r,afterany:$j --time=00:40:00 --export=ALL,RUN=$RUN,DATA="$DATA",OUT=tune_${LABEL}_noise_$n,EXTRA="$TUNE --model-noise $n" cluster/saidie/da_study.slurm)
  echo "  tuning noise $n: $j"
done
t=$(sbatch --parsable --dependency=afterany:$j --export=ALL,RUN=$RUN,DATA="$DATA",LABEL=$LABEL,NOISES="$NOISES" cluster/saidie/da_tuned.slurm)
echo "  test study at the chosen noise: $t"
