#!/bin/bash
# SM4c: fixing SM4b's 60-min overshoot (samples with ~2x the truth's 4-16 km condensate variance and a cold pool
# 0.8 K too cold at 60 min, although CRPS and calibration match the noise-channel model). All arms start from
# sm4b_flow_stagec and are scored to 90 min (train_flow.py: skill, CRPS, spread/error; smoothing_diag.py: band
# variance, spectra and extremes of single samples at 5 / 30 / 60 / 90 min). ONE GPU, sequential.
#   no retraining:  T0.5 / T0.7 / T0.85 (source-noise temperature), E8 (8 Euler steps instead of the trained 4),
#                   T1.0 (SM4b itself, the reference at 90 min)
#   retraining:     k90 (Stage C rollouts to 90 min), es (CRPS + energy score, k 45), k90es (both); 10 000 steps
#   ssh saidie 'cd ~/latent_DA && AFTER=<job> bash -l cluster/saidie/sm4c_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
dep=${AFTER:+--dependency=afterany:$AFTER}
j=$(sbatch --parsable ${dep:-} --account=chen-saidie --partition=general --gres=gpu:h200:1 --cpus-per-task=16 \
  --mem=200G --time=12:00:00 --job-name=cm1sm4c --output=logs/cm1sm4c_%j.out --wrap "set -e
[ -n \"\${CUDA_VISIBLE_DEVICES:-}\" ] || { echo 'CUDA_VISIBLE_DEVICES not set'; exit 1; }
. \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16
R=\$HOME/cm1_runs; D=\$HOME/cm1_da
DATA=cache:\$HOME/cm1_cache+\$HOME/cm1_cache_A2+\$HOME/cm1_cache_B
python -m pytest -q tests/unit/test_cm1_flow.py -p no:cacheprovider
TF=\"python scripts/cm1/train_flow.py --run \$R/full_v1_stagec --stoch \$R/full_v1_stagec_crps --source \$DATA --init-flow \$R/sm4b_flow_stagec --eval-lead-min 90 --device cuda\"
SM0=\"python scripts/cm1/smoothing_diag.py --run \$R/full_v1_stagec --source \$DATA --leads-min 5 30 60 90 --t0-min 60 120 --device cuda\"
for T in 1.0 0.85 0.7 0.5; do
  \$TF --out \$R/sm4c_T\$T --stage-c-steps 0 --temperature \$T
  \$SM0 --run-stoch \$R/sm4b_flow_stagec --temperature \$T --out \$D/sm0x_sm4c_T\$T
done
\$TF --out \$R/sm4c_E8 --stage-c-steps 0 --n-flow-eval 8
\$SM0 --run-stoch \$R/sm4b_flow_stagec --n-flow 8 --out \$D/sm0x_sm4c_E8
C=\"--stage-c-steps 10000 --stage-c-crps 4 --stage-c-batch 6 --w-sfc-th 2\"
\$TF --out \$R/sm4c_k90 \$C --stage-c-k 90
\$SM0 --run-stoch \$R/sm4c_k90 --out \$D/sm0x_sm4c_k90
\$TF --out \$R/sm4c_es \$C --stage-c-k 45 --stage-c-energy 1.0
\$SM0 --run-stoch \$R/sm4c_es --out \$D/sm0x_sm4c_es
\$TF --out \$R/sm4c_k90es \$C --stage-c-k 90 --stage-c-energy 1.0
\$SM0 --run-stoch \$R/sm4c_k90es --out \$D/sm0x_sm4c_k90es")
echo "sm4c: $j"
