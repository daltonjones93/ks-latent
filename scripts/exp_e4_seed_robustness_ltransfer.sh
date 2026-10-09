#!/bin/zsh
# E4 (docs/ks_local_field_ltransfer.tex): is Section 224's L-transfer
# result robust to training seed, or is it a property of one lucky
# checkpoint? Exact repeat of Section 224's own recipe
# (scripts/section224_ks_localfield_d48_localmlp.sh: local_field
# n_sites=16/local_channels=3, local_mlp propagator n_tokens=16/
# attn_window=3, Section 52's plain regularizers, k_pred=2,
# --full-propagator, --amp, Stage 1 200 epochs + Stage 2 k_max=12/300
# epochs) parameterized by --seed, followed by the SAME Phase F2 transfer
# check (scripts/run_ltransfer_test.py, D_KY/L at L=100 vs. transferred
# L=200) used to produce docs/RESULTS.md's original single-seed number.
#
# Usage: ./scripts/exp_e4_seed_robustness_ltransfer.sh <seed>
#
# Run once per seed (seed=0 IS Section 224 itself, already have that
# result -- re-running it here under a fresh tag keeps the comparison
# apples-to-apples rather than reusing the original artifact, in case
# anything about the pipeline has drifted since 2026-09-24/25). Each
# invocation is a full ~200+300 epoch training run -- launch seeds
# SEQUENTIALLY (one at a time, per user direction), not concurrently,
# to avoid contending for the same MPS device.
set -e
cd /Users/daltonjones/Documents/latent_DA

SEED=${1:?"usage: exp_e4_seed_robustness_ltransfer.sh <seed>"}
DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=exp_e4_seed${SEED}_localfield_d48_localmlp

echo "=== [E4 seed=$SEED] [1/5] Stage 1 ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder local_field --local-field-n-sites 16 --local-field-channels 3 \
  --aux-backbone local_mlp --mode markovian \
  --aux-n-tokens 16 --attn-window 3 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --full-propagator --amp --seed "$SEED" \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [E4 seed=$SEED] [2/5] Stage 2 ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp --seed "$SEED" \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [E4 seed=$SEED] [3/5] D_KY at L=100 (trained size) ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$STAGE2_PROP')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent

with h5py.File('$DATASET','r') as f:
    trajectories = torch.tensor(f['trajectories'][:60], dtype=torch.float32)
    traj3 = trajectories[53]

with torch.no_grad():
    z01 = ae.encode(traj3[:2]).numpy()
res = lyapunov_spectrum_latent_propagator(
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=2000, qr_every=10,
    dt_snap=1.0, warmup_steps=200, seed=0, max_abs_state=1e4,
)
print('[E4 seed=$SEED] L=100 D_KY:', res.kaplan_yorke_dimension, ' lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', d_latent)
with open('artifacts/experiments/e4_seed${SEED}_dky_L100.txt', 'w') as f:
    f.write(str(res.kaplan_yorke_dimension))
" 2>&1 | tee artifacts/logs/e4_seed${SEED}_dky_L100.log

mkdir -p artifacts/experiments
D_KY_100=$(cat artifacts/experiments/e4_seed${SEED}_dky_L100.txt)

echo "=== [E4 seed=$SEED] [4/5] Phase F2 transfer check: L=100 -> L=200, zero retraining ==="
mamba run -n da_env python scripts/run_ltransfer_test.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --old-L 100.0 --new-L 200.0 --old-D-KY "$D_KY_100" --dt-snap 1.0 --seed 0 \
  > artifacts/logs/e4_seed${SEED}_ltransfer_L200.log 2>&1
cat artifacts/logs/e4_seed${SEED}_ltransfer_L200.log

echo "=== [E4 seed=$SEED] [5/5] complete ==="
