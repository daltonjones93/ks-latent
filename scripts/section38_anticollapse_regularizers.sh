#!/bin/zsh
# User-directed 2026-08-31: "try a couple of stage 1 runs for 120 epochs
# using the two new regularizers we developed. make their weight .04 in
# each case... Use the same encoder decoder and aux propagator model as
# [stage2_multiseed_wvmablation-wvm00_seed0.log]. stage 2 of training in
# both of these cases should use a markovian mlp, w_varmatch = 0.
# k_max = 12."
#
# "Same encoder decoder and aux propagator model" = the same architecture
# that AE's whole lineage was built from -- the canonical
# history3_fullprop_wvar005_tw16_wspatial005 recipe (vit encoder,
# vit/history aux, w_var=0.05, w_spatial=0.05, full_propagator,
# prop_delta_cap=0.5) -- trained FRESH (not from that already-collapsed
# checkpoint's weights), since the whole point is testing whether these
# regularizers prevent the collapse from happening in the first place.
#
# gamma/eps left at variance_floor_loss's/logdet_barrier_loss's own
# documented defaults (var_floor_gamma=0.1, logdet_eps=1e-3) -- both
# already reasoned through in their own docstrings; only the requested
# top-level weight (0.04) varies between the two runs.
set -e
cd /Users/daltonjones/Documents/latent_DA

BASE_RECIPE=(--profile full --encoder vit --aux-backbone vit --mode history --n-history 3 \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --aux-n-tokens 44 --aux-token-d-model 64 --w-var 0.05 \
  --full-propagator --prop-delta-cap 0.5 --w-spatial 0.05 \
  --epochs 120 --checkpoint-every 20)

STAGE2_RECIPE=(--backbone mlp --mode markovian \
  --epochs 100 --k-max 12 --k-warmup-epochs 71 --k-mid 8 --k-mid-epochs 59)

check_spectrum() {
  local label=$1
  local ae_path=$2
  mamba run -n da_env python -c "
import h5py, numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint
with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj.shape
ae, cfg, _ = load_autoencoder_checkpoint('$ae_path')
ae.eval()
with torch.no_grad():
    z = ae.encode(traj.reshape(n*T, NX)).reshape(n*T, -1).numpy()
cov = np.cov(z, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print('$label: min_eig=%.4e  cond#=%.4e  top_eig=%.3f' % (eig[-1], eig[0]/eig[-1], eig[0]))
print('  bottom 10:', np.array2string(eig[-10:], formatter={'float_kind':lambda x: f'{x:.2e}'}))
"
}

echo "=== [Run A 1/2] Stage 1: variance-floor regularizer (w_var_floor=0.04), 120 epochs ==="
TAG_A=history3_fullprop_wvar005_tw16_wspatial005_varfloor04_120ep
mamba run -n da_env python scripts/train_stage1_patched.py \
  "${BASE_RECIPE[@]}" --w-var-floor 0.04 \
  --tag "$TAG_A" \
  > artifacts/logs/stage1_${TAG_A}.log 2>&1
AE_A=artifacts/stage1_ae_patched_full_${TAG_A}.pt
echo "=== [Run A 2/2] Latent covariance spectrum check ==="
check_spectrum "variance-floor (w=0.04)" "$AE_A" > artifacts/logs/spectrum_${TAG_A}.log 2>&1
cat artifacts/logs/spectrum_${TAG_A}.log

echo "=== [Run A] Stage 2: fresh mlp/markovian, w_varmatch=0, k_max=12 ==="
STAGE2_TAG_A="${TAG_A}_mlpmarkovian_k12"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE_A" "${STAGE2_RECIPE[@]}" \
  --tag "$STAGE2_TAG_A" \
  > artifacts/logs/stage2_${STAGE2_TAG_A}.log 2>&1

echo "=== [Run B 1/2] Stage 1: log-det barrier regularizer (w_logdet=0.04), 120 epochs ==="
TAG_B=history3_fullprop_wvar005_tw16_wspatial005_logdet04_120ep
mamba run -n da_env python scripts/train_stage1_patched.py \
  "${BASE_RECIPE[@]}" --w-logdet 0.04 \
  --tag "$TAG_B" \
  > artifacts/logs/stage1_${TAG_B}.log 2>&1
AE_B=artifacts/stage1_ae_patched_full_${TAG_B}.pt
echo "=== [Run B 2/2] Latent covariance spectrum check ==="
check_spectrum "log-det barrier (w=0.04)" "$AE_B" > artifacts/logs/spectrum_${TAG_B}.log 2>&1
cat artifacts/logs/spectrum_${TAG_B}.log

echo "=== [Run B] Stage 2: fresh mlp/markovian, w_varmatch=0, k_max=12 ==="
STAGE2_TAG_B="${TAG_B}_mlpmarkovian_k12"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE_B" "${STAGE2_RECIPE[@]}" \
  --tag "$STAGE2_TAG_B" \
  > artifacts/logs/stage2_${STAGE2_TAG_B}.log 2>&1

echo "=== Section 38 (both anti-collapse regularizer runs) complete ==="
echo "--- Run A (variance floor) ---"
tail -3 artifacts/logs/stage2_${STAGE2_TAG_A}.log
echo "--- Run B (log-det barrier) ---"
tail -3 artifacts/logs/stage2_${STAGE2_TAG_B}.log
