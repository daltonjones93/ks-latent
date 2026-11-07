#!/bin/bash
# SM4b: the flow-matching propagator with rollout training. Velocity network pointwise in x (--flow-radius 0:
# reach 3 sites/step as the stencil propagator), 4 Euler steps; one-step flow-matching pretraining on
# full_v1_stagec's latents, then Stage C exactly as the stochastic model's (decoded 45-step rollouts, 4
# samples, fair CRPS, batch 6, 15 000 steps, near-surface theta' weight 2), autoencoder updated jointly.
# Scored against full_v1_stagec (deterministic) and full_v1_stagec_crps (stochastic), held boundaries.
#   ssh saidie 'cd ~/latent_DA && AFTER=<job> bash -l cluster/saidie/sm4b_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
R=$HOME/cm1_runs
dep=${AFTER:+--dependency=afterany:$AFTER}
j=$(sbatch --parsable ${dep:-} --account=chen-saidie --partition=general --gres=gpu:h200:1 --cpus-per-task=16 \
  --mem=200G --time=06:00:00 --job-name=cm1sm4b --output=logs/cm1sm4b_%j.out --wrap "set -e
[ -n \"\${CUDA_VISIBLE_DEVICES:-}\" ] || { echo 'CUDA_VISIBLE_DEVICES not set'; exit 1; }
. \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16
python -m pytest -q tests/unit/test_cm1_flow.py -p no:cacheprovider
python scripts/cm1/train_flow.py --run $R/full_v1_stagec --stoch $R/full_v1_stagec_crps --source $DATA \
  --out $R/sm4b_flow_stagec --n-flow 4 --flow-radius 0 --stage-c-steps 15000 --stage-c-k 45 --stage-c-crps 4 \
  --stage-c-batch 6 --w-sfc-th 2 --device cuda")
echo "sm4b_flow_stagec: $j"
