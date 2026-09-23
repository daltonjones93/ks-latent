#!/bin/zsh
# User-directed 2026-09-02, "Section 61": follow-up to Section 60, whose
# w_spatial(signed) decay started AT 0.08 -- the exact weight that caused
# Section 47's catastrophic collapse (cond#=1.07e7, D_KY crashed to
# 7.97). Section 60 landed at cond#=8.53e6 and val_kmax_mse=0.0482, both
# far worse than Section 52's fixed-weight baseline (1.59e6 / 0.0256),
# consistent with that early-collapse damage not being reversible by the
# later low-weight epochs (this project's collapse is correlation-driven,
# not just variance-shrinkage). This run keeps the same w_var/w_logdet
# decays as Section 60 but starts w_spatial(signed) at a known-safe value
# instead: 0.025 -> 0.002 (vs. Section 60's 0.08 -> 0.004). Stage 1 also
# reverts to 200 epochs (Section 60 used 250 by request; the 250-epoch
# change was not implicated in the failure, but the user asked to revert
# it here). Otherwise identical to Section 52's recipe (default-size vit
# AE + mlp/markovian aux, w_decorr=0, w_var_floor=0, NO delta_cap, --amp,
# Stage 2 warm-started, k_max=12, Stage 2 300ep). w_pred stays at
# Stage1TrainingConfig's default (0.5), unchanged.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section61_mlpmarkovian_wvardecay005to001_wspatialsigneddecay0025to0002_logdetdecay002to00005_200ep

echo "=== [1/3] Stage 1: mlp/markovian aux, w_decorr=0, w_var 0.05->0.01, w_spatial(signed) 0.025->0.002, w_var_floor=0, w_logdet 0.02->0.0005, NO delta_cap, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone mlp --mode markovian \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --w-decorr 0 \
  --w-var 0.05 --w-var-end 0.01 \
  --w-spatial 0.025 --w-spatial-end 0.002 --spatial-signed \
  --w-var-floor 0 \
  --w-logdet 0.02 --w-logdet-end 0.0005 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Latent covariance spectrum + D7/D8 bandedness check ==="
mamba run -n da_env python -c "
import h5py, numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint
from ks_latent.analysis.diagnostics import same_time_coupling_diagnostic, same_time_coupling_diagnostic_signed
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
d7 = same_time_coupling_diagnostic(z_np, n_null=500, seed=0)
print('D7 bandedness=%.4f p=%.4f' % (d7.bandedness_observed, d7.bandedness_p_value))
d8 = same_time_coupling_diagnostic_signed(z_np, n_null=500, seed=0)
print('D8 signed bandedness=%.4f p=%.4f' % (d8.bandedness_observed, d8.bandedness_p_value))
l_spatial_unsigned = spatial_coherence_loss(z, bandwidth=3.0, signed=False)
l_spatial_signed = spatial_coherence_loss(z, bandwidth=3.0, signed=True)
print('l_spatial (unsigned) raw=%.6f (lower=more coherent)' % l_spatial_unsigned.item())
print('l_spatial (signed) raw=%.6f (lower=more coherent)' % l_spatial_signed.item())
" > artifacts/logs/spectrum_${TAG}.log 2>&1
cat artifacts/logs/spectrum_${TAG}.log

echo "=== [3/3] Stage 2: warm-started from Stage 1's own mlp/markovian aux, NO delta_cap, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 61 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
