#!/bin/zsh
# User-directed 2026-09-02, "Section 60": test whether decaying Stage 1's
# anti-collapse regularizer weights over training (high early, low late)
# recovers the DA/rollout performance lost when those weights are held
# high throughout (Sections 56 vs. 58) while still getting most of the
# early-training conditioning benefit. Same recipe as Section 52
# (default-size vit AE + mlp/markovian aux, w_decorr=0, w_var_floor=0, NO
# delta_cap, --amp, Stage 2 warm-started, k_max=12, Stage 2 300ep)
# EXCEPT Stage 1 now runs 250 epochs (up from 200) with three NEW linear
# decays (ks_latent.training.loops.linear_decay, wired via
# Stage1TrainingConfig.w_var_end/w_logdet_end/w_spatial_end):
#   w_var:            0.05  -> 0.01   (--w-var 0.05 --w-var-end 0.01)
#   w_logdet:         0.02  -> 0.0005 (--w-logdet 0.02 --w-logdet-end 0.0005)
#   w_spatial(signed): 0.08 -> 0.004  (--w-spatial 0.08 --w-spatial-end 0.004)
# All three decay linearly from epoch 0 to the final epoch (249) --
# floors are NONZERO deliberately (not decayed to 0): this project's
# latent collapse is correlation-driven, not just variance-shrinkage, so
# fully removing anti-collapse pressure late in training risks drifting
# back toward collapse right when regularization is weakest. w_pred
# stays at Stage1TrainingConfig's default (0.5), unchanged -- confirmed
# directly from config.py, Section 52 never overrode it either.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section60_mlpmarkovian_wvardecay005to001_wspatialsigneddecay008to0004_logdetdecay002to00005_250ep

echo "=== [1/3] Stage 1: mlp/markovian aux, w_decorr=0, w_var 0.05->0.01, w_spatial(signed) 0.08->0.004, w_var_floor=0, w_logdet 0.02->0.0005, NO delta_cap, --amp, 250 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone mlp --mode markovian \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --w-decorr 0 \
  --w-var 0.05 --w-var-end 0.01 \
  --w-spatial 0.08 --w-spatial-end 0.004 --spatial-signed \
  --w-var-floor 0 \
  --w-logdet 0.02 --w-logdet-end 0.0005 \
  --full-propagator --amp \
  --epochs 250 --checkpoint-every 20 \
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

echo "=== Section 60 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
