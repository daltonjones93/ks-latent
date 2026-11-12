#!/bin/bash -l
# 5v: latent DA vs ensemble size for one (pruned) model, G0 protocol per (model, N): reflectivity head, noise tuned on the 6 validation
# storms (--tuning), then the 22 test storms x 3 seeds at the chosen noise. One GPU, sequential.
#   sbatch --export=ALL,RUN=fpc_nested_p8_dft,LABEL=p8,NS="8 16 32 64",NOISES="0.05 0.1 0.15" cluster/saidie/da_prune.slurm
#SBATCH --job-name=daprune
#SBATCH --account=chen-saidie
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/daprune_%j.out
set -euo pipefail
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES not set: refusing to run on an unassigned GPU"; exit 1; }
REPO=$HOME/latent_DA
. "${ENV:-$HOME/venvs/cm1}/bin/activate"
export PYTHONPATH=$REPO OMP_NUM_THREADS=8
cd "$REPO"
RUN=${RUN:?} LABEL=${LABEL:?} NS=${NS:-"8 16 32 64"} NOISES=${NOISES:-"0.05 0.1 0.15"}
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
[ -f "$HOME/cm1_runs/$RUN/refl_head.pt" ] || python scripts/cm1/train_refl_head.py --run "$HOME/cm1_runs/$RUN" --source "$DATA" --device cuda
COMMON="--modes latent --random-members --seeds 0 1 2 --obs-set th+refl+vr --boundary persistence"
for N in $NS; do
  dirs=""
  for n in $NOISES; do
    out=$HOME/cm1_da/prune_${LABEL}_N${N}_tune_$n
    python scripts/cm1/da_study.py --run "$HOME/cm1_runs/$RUN" --source "$DATA" --out "$out" --device cuda --tuning $COMMON --n-members $N --model-noise $n
    dirs="$dirs $out"
  done
  noise=$(python scripts/cm1/pick_tuned.py --record "$HOME/cm1_da/prune_${LABEL}_N${N}_choice.json" $dirs | tee /dev/stderr | tail -1)
  echo "N=$N chosen on validation: model noise $noise"
  python scripts/cm1/da_study.py --run "$HOME/cm1_runs/$RUN" --source "$DATA" --out "$HOME/cm1_da/prune_${LABEL}_N${N}" --device cuda \
      --include-test $COMMON --n-members $N --model-noise "$noise"
done
