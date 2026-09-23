#!/bin/zsh
# User-directed 2026-08-31 (revised from section39): "I think we should
# just lower w_logdet honestly, it changed the condition number hugely
# before. kill the current run, and start it again, with w_spatial = .05
# and w_logdet = .005. just run for 120 epochs. then proceed to stage 2
# as before."
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=history3_fullprop_wvar005_tw16_wspatial005_logdet0005_120ep

echo "=== [1/4] Stage 1: w_spatial=0.05 (back to original), w_logdet=0.005 (down from 0.04), 120 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone vit --mode history --n-history 3 \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --aux-n-tokens 44 --aux-token-d-model 64 --w-var 0.05 \
  --full-propagator --prop-delta-cap 0.5 \
  --w-spatial 0.05 --w-logdet 0.005 \
  --epochs 120 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt

echo "=== [2/4] Latent covariance spectrum + D7 bandedness check ==="
mamba run -n da_env python -c "
import h5py, numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint
from ks_latent.analysis.diagnostics import same_time_coupling_diagnostic
from ks_latent.training.losses import spatial_coherence_loss, logdet_barrier_loss

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
l_spatial = spatial_coherence_loss(z, bandwidth=3.0)
l_logdet = logdet_barrier_loss(z, eps=1e-3)
print('l_spatial raw=%.6f (lower=more coherent)' % l_spatial.item())
print('l_logdet raw=%.6f' % l_logdet.item())
" > artifacts/logs/spectrum_${TAG}.log 2>&1
cat artifacts/logs/spectrum_${TAG}.log

echo "=== [3/4] Stage 2: fresh mlp/markovian, w_varmatch=0, k_max=12 (unchanged) ==="
STAGE2_TAG="${TAG}_mlpmarkovian_k12"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone mlp --mode markovian \
  --epochs 100 --k-max 12 --k-warmup-epochs 71 --k-mid 8 --k-mid-epochs 59 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== [4/4] Section 39b complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
