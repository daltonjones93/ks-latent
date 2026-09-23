#!/bin/zsh
# User-directed 2026-09-01, "Section 42": following the terms-audit
# discussion, a deliberately different recipe testing several changes at
# once:
#  Stage 1: eliminate L_decorr and L_var (w_decorr=0, w_var=0), raise
#    w_spatial to 0.08, raise w_pred to 1.0, and enable the existing
#    multi-step k_pred_max rollout mechanism ("add an extra rollout so we
#    iteratively roll out the latent dimension several times using the
#    aux propagator and measure loss against future decoded states").
#    Aux propagator changed from vit/history to mlp/markovian specifically
#    so it can be warm-started directly into Stage 2 (same architecture
#    family).
#  Stage 2: same standard recipe as Sections 39/40 (k_max=12, 100 epochs),
#    but WARM-STARTED from Stage 1's own mlp/markovian aux propagator
#    (not fresh), plus a NEW w_spatial=0.005 term (ks_latent/config.py's
#    Stage2TrainingConfig.w_spatial, ks_latent/training/loops.py's
#    train_stage2 -- newly implemented today, applies spatial_coherence_loss
#    to the propagator's own rolled-out predictions).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section42_mlpaux_wspatial08_wpred1_ms4_logdet0005

echo "=== [1/3] Stage 1: mlp/markovian aux, w_decorr=0, w_var=0, w_spatial=0.08, w_pred=1.0, w_logdet=0.005, k_pred_max=4, 120 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone mlp --mode markovian \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --w-decorr 0 --w-var 0 --w-spatial 0.08 --w-pred 1.0 --w-logdet 0.005 \
  --full-propagator --prop-delta-cap 0.5 \
  --k-pred-max 4 \
  --epochs 120 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Latent covariance spectrum + D7 bandedness check ==="
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
print('l_spatial raw=%.6f (lower=more coherent)' % l_spatial.item())
" > artifacts/logs/spectrum_${TAG}.log 2>&1
cat artifacts/logs/spectrum_${TAG}.log

echo "=== [3/3] Stage 2: warm-started from Stage 1's own mlp/markovian aux, + w_spatial=0.005, k_max=12, 100 epochs ==="
STAGE2_TAG="${TAG}_warmstart_wspatial0005_k12"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --w-spatial 0.005 \
  --epochs 100 --k-max 12 --k-warmup-epochs 71 --k-mid 8 --k-mid-epochs 59 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 42 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
