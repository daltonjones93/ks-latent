#!/bin/zsh
# User-directed 2026-09-01, "Section 46": same recipe family as Sections
# 44/45 (w_decorr=0, w_var_floor=0, NO delta_cap, Stage 2 warm-started,
# k_max=12), but w_spatial=0.08, w_var=0.04, w_logdet=0.002, Stage 1 for
# 200 epochs, Stage 2 kept at Section 44's original 300 epochs
# ("everything else the same" -- no Stage-2 epoch override given). Also
# the first real test of --amp (bfloat16 autocast, implemented earlier
# today) on a full run, per the standing instruction to use --amp for all
# future runs going forward -- caught and fixed a real bug along the way
# (torch.linalg.slogdet doesn't support bfloat16 at all, unlike softmax/
# layer_norm which autocast upcasts automatically; logdet_barrier_loss
# now explicitly casts to float32 itself). Aux propagator changed from
# mlp/history(n_history=2) to mlp/MARKOVIAN per a follow-up user request
# (mid-run, before the first epoch-10 timing readout came back).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section46_mlpmarkovian_wvar004_wspatial008_logdet0002_200ep

echo "=== [1/3] Stage 1: mlp/markovian aux, w_decorr=0, w_var=0.04, w_spatial=0.08, w_var_floor=0, w_logdet=0.002, NO delta_cap, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone mlp --mode markovian \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --w-decorr 0 --w-var 0.04 --w-spatial 0.08 --w-var-floor 0 --w-logdet 0.002 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own mlp/history(n=2) aux, NO delta_cap, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 46 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
