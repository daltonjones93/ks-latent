#!/bin/zsh
# User-directed 2026-09-01, "Section 44": Stage 1 with w_recon=1.0 (default),
# w_pred=0.5 (default), w_decorr=0, w_var=0.03, w_spatial=0.1,
# w_var_floor=0, w_logdet=0.005. Aux propagator = mlp/history(n_history=2,
# current + 1 past state) -- the SAME propagator warm-started directly
# into Stage 2 (matching Section 42's original intent, this time avoiding
# its delta_cap warm-start confound: delta_cap is OFF in both stages,
# never passed at all, so there is nothing for warm-starting to
# accidentally carry over). Stage 2 runs for 300 epochs.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section44_mlphistory2_wvar003_wspatial01_logdet0005_nodeltacap

echo "=== [1/3] Stage 1: mlp/history(n=2) aux, w_decorr=0, w_var=0.03, w_spatial=0.1, w_var_floor=0, w_logdet=0.005, NO delta_cap, 120 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone mlp --mode history --n-history 2 \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --w-decorr 0 --w-var 0.03 --w-spatial 0.1 --w-var-floor 0 --w-logdet 0.005 \
  --full-propagator \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own mlp/history(n=2) aux, NO delta_cap, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 44 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
