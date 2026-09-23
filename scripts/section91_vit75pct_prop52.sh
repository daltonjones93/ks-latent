#!/bin/zsh
# User-directed 2026-09-05, "Section 91": "can we keep the same encoder
# and decoder, but inflate the number of parameters in the vit part of
# the networks and decrease the parameters in the fourier_mlp part of the
# networks. let's have it a 75% of the parameters come from the ViT for
# each. Then maybe we just use the same propagator as 52. keep the same
# regularizer parameters as 90."
#
# Base: Section 90's exact encoder/decoder STRUCTURE (vit_fourier_hybrid,
# ViT pool=local/pool_window=8 encoder + pool=mean decoder,
# fourier_kind=ifft, enc_out_modes=8, dec_fno_modes=12, pos_encoding=linear,
# attn_window=4, token_window=16) -- UNCHANGED except the ViT/Fourier
# capacity split, rebalanced to 75%/25% (was ~50/50 in Section 90):
#   - d_model: 92 -> 116 (ViT branch, both encode+decode share this width)
#   - fourier_hidden: 270 -> 185 (Fourier branch, both encode+decode share
#     this width; fourier_blocks stays 2)
# Verified this turn by direct instantiation sweep: vit=1,149,581 (75.1%),
# fourier=381,004 (24.9%), total=1,530,585 -- essentially matches Section
# 90's own total (1,492,299), so this isolates the RATIO as the only size
# variable, not overall scale.
#
# Propagator: Section 52's EXACT recipe -- --aux-backbone mlp --mode
# markovian, default sizing under --full-propagator (hidden=128,
# n_blocks=3, no --aux-hidden/--aux-blocks override) -- REPLACES Section
# 89/90's fourier_mlp/history2 propagator (~1M params) entirely. This is
# a genuinely new combination: a much smaller (~111K-param) propagator
# paired with a considerably more complex, differently-organized latent
# space (local-pooled + Fourier-hybrid encoder) than Section 52's own
# plain dense-ViT encoder ever produced -- a direct test of the
# propagator-capacity confound flagged since Section 34/78 but never
# isolated (see docs/model-research-summary-9-5-26.md's Section 34/78
# discussion).
#
# Regularizers: Section 90's exact recipe (w_var=0.02, w_spatial=0.019
# SIGNED, w_var_floor=0, w_logdet=0.008, w_smooth=0.0005, NO lambda_z,
# w_decorr=0) -- unchanged.
#
# EXPECTED RESULT (stated before launch, per user request): reconstruction
# should stay good (more ViT capacity); rollout accuracy (val_kmax_mse) is
# expected to come in WORSE than both plain Section 52 (0.0256) and
# Section 90 (0.0346) -- w_spatial=0.019 already regressed Section 87
# (built on 52's own plain architecture) to 0.0351, and pairing a much
# smaller propagator with a more complex latent space than 52's own is an
# untested combination that could compound that regression. Smoothness
# (step size/curvature) is expected to stay good, independent of
# propagator choice (driven by the encoder's local pooling/enc_out_modes
# truncation, both unchanged from 90). D_KY should still land near the
# true ~22 benchmark regardless (mlp/markovian is global+unconstrained,
# satisfying the project's core propagator-collapse-avoidance mechanism).
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.035034, no errors) and a 4-epoch Stage 2 warm-start smoke run
# (val_kmax_mse trending down 0.316->0.212 by epoch 3) under this exact
# flag combination. No code changed this turn (pure hyperparameter/
# propagator-choice combination of existing capabilities). Smoke
# artifacts cleaned up before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section91_vitfourieriffthybrid_dmodel116_fourierhidden185_75pctvit_propmlpmarkovian52_wspatial019_wsmooth0005_200ep

echo "=== [1/3] Stage 1: vit_fourier_hybrid encoder (pool=local/w8, enc_out_modes=8) / decoder (pool=mean, dec_fno_modes=12) rebalanced to 75% ViT (d_model=116) / 25% Fourier (fourier_hidden=185) + Section 52's EXACT mlp/markovian propagator (hidden=128/n_blocks=3, default full-propagator sizing), w_decorr=0, w_var=0.02, w_spatial=0.019 (SIGNED), w_var_floor=0, w_logdet=0.008, w_smooth=0.0005, NO lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit_fourier_hybrid --aux-backbone mlp --mode markovian \
  --d-model 116 --pos-encoding linear --attn-window 4 --token-window 16 --pool local --pool-window 8 --dec-pool mean \
  --vit-fourier-fourier-hidden 185 --vit-fourier-fourier-blocks 2 --vit-fourier-kind ifft \
  --vit-fourier-enc-out-modes 8 \
  --fourier-mlp-dec-fno-modes 12 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.019 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.0005 \
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
print('participation ratio: %.3f (out of d_latent=%d)' % (eig.sum()**2 / (eig**2).sum(), cfg.d_latent))
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own mlp/markovian aux, --amp, k_max=12, 300 epochs (Section 52's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 91 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
