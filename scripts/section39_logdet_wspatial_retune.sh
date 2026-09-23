#!/bin/zsh
# User-directed 2026-08-31: "it looks like the logdet run doesn't have
# quite as much spatial coherence. can we up wspatial to .065, lower the
# log det to .025, and run it for 200 epochs. keep everything the same
# including the stage 2 run."
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=history3_fullprop_wvar005_tw16_wspatial065_logdet025_200ep

echo "=== [1/4] Stage 1: w_spatial=0.065 (up from 0.05), w_logdet=0.025 (down from 0.04), 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone vit --mode history --n-history 3 \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --aux-n-tokens 44 --aux-token-d-model 64 --w-var 0.05 \
  --full-propagator --prop-delta-cap 0.5 \
  --w-spatial 0.065 --w-logdet 0.025 \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt

echo "=== [2/4] Latent covariance spectrum + D7 bandedness check ==="
mamba run -n da_env python -c "
import h5py, numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint
from ks_latent.analysis.diagnostics import same_time_coupling_diagnostic

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj.shape
ae, cfg, _ = load_autoencoder_checkpoint('$AE')
ae.eval()
with torch.no_grad():
    z = ae.encode(traj.reshape(n*T, NX))
z_np = z.numpy()
cov = np.cov(z_np, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print('spectrum: min_eig=%.4e  cond#=%.4e  top_eig=%.3f' % (eig[-1], eig[0]/eig[-1], eig[0]))
print('bottom 10:', np.array2string(eig[-10:], formatter={'float_kind':lambda x: f'{x:.2e}'}))
result = same_time_coupling_diagnostic(z_np, n_null=500, seed=0)
print('D7 bandedness=%.4f p=%.4f' % (result.bandedness_observed, result.bandedness_p_value))
" > artifacts/logs/spectrum_${TAG}.log 2>&1
cat artifacts/logs/spectrum_${TAG}.log

echo "=== [3/4] Stage 2: fresh mlp/markovian, w_varmatch=0, k_max=12 (unchanged from Section 38) ==="
STAGE2_TAG="${TAG}_mlpmarkovian_k12"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone mlp --mode markovian \
  --epochs 100 --k-max 12 --k-warmup-epochs 71 --k-mid 8 --k-mid-epochs 59 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== [4/4] Section 39 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
