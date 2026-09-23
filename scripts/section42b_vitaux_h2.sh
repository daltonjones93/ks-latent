#!/bin/zsh
# User-directed 2026-09-01: "there seems to be something about the vit as
# aux propagator that limits the instability of the resulting condition
# number for the covariance matrix. can we try this same stage 1 run as
# before but use the vit backbone with history 2. then keep the stage 2
# training the same." Testing whether the mlp/markovian aux (Section 42's
# choice, which produced an anomalous top_eigenvalue=51.04, ~10x every
# other run today) was itself the cause of that anomaly and the
# catastrophic warm-started Stage 2 result, by swapping back to
# vit/history (n_history=2, not the usual 3) with everything else
# unchanged. vit/history aux is architecturally incompatible with
# mlp/markovian warm-starting, so Stage 2 uses a FRESH mlp/markovian
# propagator instead -- everything else (w_spatial=0.005, k_max=12, 100
# epochs) kept exactly as Section 42's original Stage 2 spec.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section42b_vitaux_h2_wspatial08_wpred1_ms4_logdet0005

echo "=== [1/3] Stage 1: vit/history(n=2) aux, w_decorr=0, w_var=0, w_spatial=0.08, w_pred=1.0, w_logdet=0.005, k_pred_max=4, 120 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone vit --mode history --n-history 2 \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --aux-n-tokens 44 --aux-token-d-model 64 \
  --w-decorr 0 --w-var 0 --w-spatial 0.08 --w-pred 1.0 --w-logdet 0.005 \
  --full-propagator --prop-delta-cap 0.5 \
  --k-pred-max 4 \
  --epochs 120 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt

echo "=== [2/3] Latent covariance spectrum + D7 bandedness check ==="
mamba run -n da_env python -c "
import h5py, numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint
from ks_latent.analysis.diagnostics import same_time_coupling_diagnostic
from ks_latent.training.losses import spatial_coherence_loss

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
print('top 10:', np.array2string(eig[:10], precision=3))
print('bottom 10:', np.array2string(eig[-10:], formatter={'float_kind':lambda x: f'{x:.2e}'}))
result = same_time_coupling_diagnostic(z_np, n_null=500, seed=0)
print('D7 bandedness=%.4f p=%.4f' % (result.bandedness_observed, result.bandedness_p_value))
l_spatial = spatial_coherence_loss(z, bandwidth=3.0)
print('l_spatial raw=%.6f (lower=more coherent)' % l_spatial.item())
" > artifacts/logs/spectrum_${TAG}.log 2>&1
cat artifacts/logs/spectrum_${TAG}.log

echo "=== [3/3] Stage 2: FRESH mlp/markovian (vit/history aux can't warm-start mlp), w_spatial=0 (per user correction), k_max=12, 100 epochs ==="
STAGE2_TAG="${TAG}_freshmlp_nowspatial_k12"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone mlp --mode markovian \
  --epochs 100 --k-max 12 --k-warmup-epochs 71 --k-mid 8 --k-mid-epochs 59 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 42b complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
