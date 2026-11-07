#!/bin/bash
# Track FP, FP1: 2D latent-field autoencoders, one job on ONE GPU, data loaded once. Sweep: at the data's
# resolution, jet and pointwise decoders with m = 1, 2, 3, 4, 6 fields; on coarser latent grids
# (stride x / z = 2/1, 4/2, 8/4) the jet decoder with m = 2, 4, 8. Same split and normalizer as full_v1;
# scored on the 22 test storms like the column autoencoders, plus the rediscovery probe.
#   ssh saidie 'cd ~/latent_DA && AFTER=<job> bash -l cluster/saidie/f1_sweep.sh'
# FP1b (joint propagation): CONFIGS="..." OUT=~/cm1_runs/f1b_prop EXTRA="--prop-k 5" bash -l cluster/saidie/f1_sweep.sh
set -euo pipefail
cd "$HOME/latent_DA"
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
CONFIGS=${CONFIGS:-"1:1:1:jet 2:1:1:jet 3:1:1:jet 4:1:1:jet 6:1:1:jet 1:1:1:point 2:1:1:point 3:1:1:point 4:1:1:point 6:1:1:point 2:2:1:jet 4:2:1:jet 8:2:1:jet 2:4:2:jet 4:4:2:jet 8:4:2:jet 2:8:4:jet 4:8:4:jet 8:8:4:jet"}
OUT=${OUT:-$HOME/cm1_runs/f1_sweep}
dep=${AFTER:+--dependency=afterany:$AFTER}
j=$(sbatch --parsable ${dep:-} --account=chen-saidie --partition=general --gres=gpu:h200:1 --cpus-per-task=16 \
  --mem=200G --time=12:00:00 --job-name=cm1f1 --output=logs/cm1f1_%j.out --wrap "set -e
[ -n \"\${CUDA_VISIBLE_DEVICES:-}\" ] || { echo 'CUDA_VISIBLE_DEVICES not set'; exit 1; }
. \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16
python -m pytest -q tests/unit/test_cm1_field_ae.py -p no:cacheprovider
python scripts/cm1/train_field_ae.py --profile smoke --out $OUT/smoke --configs 2:1:1:jet 2:2:1:jet --device cuda ${EXTRA:-} | tail -4
python scripts/cm1/train_field_ae.py --profile full --ref-run $HOME/cm1_runs/full_v1 --source $DATA --out $OUT \
  --configs $CONFIGS --steps ${STEPS:-20000} --device cuda ${EXTRA:-}")
echo "f1_sweep: $j"
