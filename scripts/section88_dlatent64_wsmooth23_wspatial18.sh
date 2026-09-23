#!/bin/zsh
# User-directed 2026-09-04, "Section 88": "make section 88 use the model
# and setup of section 82, except use d_latent = 64, but use the
# regularization params from 87 except increase w_smooth by 2.3x and
# w_spatial (signed) by 1.8."
#
# Architecture: Section 82's exact setup (scripts/section82_dlatent56_more_spatial.sh)
# -- vit_fourier_hybrid AE (ViT d_model=92, Section 52 structure, SUMMED
# with FourierIFFTBody hidden=270/n_blocks=2, fourier_kind=ifft) +
# fourier_mlp/history2 propagator (hidden=480/n_blocks=2,
# fourier_ifft_readout), EXCEPT:
#   - d_latent: 56 -> 64 (82 itself had already grown this from 75/81's
#     44). --prop-attn-window MUST track d_latent (operates directly in
#     d_latent-ring space, unlike the ViT branch's --attn-window=4 which
#     is a TOKEN-space window, unaffected by d_latent) -- raised 28->32
#     (=64//2, the new "effectively fully dense" saturation point, per
#     the standing convention: 22 at d_latent=44, 28 at 56, 32 at 64).
#     Verified this turn by direct instantiation: mask is all-True at
#     window=32/d_latent=64. AE -> 1,580,998 params, propagator body ->
#     1,046,946 params.
#
# Regularizers: Section 87's recipe (w_var=0.02, w_logdet=0.004,
# w_var_floor=0, w_decorr=0, NO lambda_z -- the 52/86/87 lineage never
# uses it), with w_spatial and w_smooth further scaled up from 87's own
# values per this turn's explicit request:
#   - w_spatial (signed): 87's 0.019 -> 0.0342 (x1.8).
#   - w_smooth: 87's 0.002 -> 0.0046 (x2.3).
# w_logdet stays at 87's 0.004 (not requested to change) -- notably
# HIGHER than Section 82's own w_logdet=0.008... no, LOWER (82 used
# 0.008); kept at 87's value regardless since the user said "use the
# params from 87" for everything not explicitly further scaled.
#
# Curriculum/other flags identical to 82: --pool mean, --pos-encoding
# linear, --attn-window 4, --token-window 16, --full-propagator, --amp,
# 200ep Stage 1, Stage-2 warm-started k_max=12 curriculum
# (k_warmup_epochs=210/k_mid=8/k_mid_epochs=175), 300ep Stage 2.
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.009468, no errors) and a 4-epoch Stage 2 warm-start smoke run
# (val_kmax_mse trending down 0.20->0.089 by epoch 3, a healthy early
# trajectory) under this exact flag combination. No code changed this
# turn. Smoke artifacts cleaned up before this real launch.
#
# Queued behind Section 87 (outer PID chain from
# scripts/section87_vit52_wsmooth2.sh) to avoid MPS training contention
# -- launched automatically once that finishes.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section88_dlatent64_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_wspatial0342_wsmooth0046_logdet004_200ep

echo "=== [1/3] Stage 1: vit_fourier_hybrid encoder/decoder (d_latent=64, ViT d_model=92 + FourierIFFTBody hidden=270/n_blocks=2, ifft kind) + fourier_mlp propagator (d_latent=64, attn_window=32, fourier_ifft_readout, hidden=480/n_blocks=2), w_decorr=0, w_var=0.02, w_spatial=0.0342 (SIGNED, =87's 0.019 x1.8), w_var_floor=0, w_logdet=0.004 (=87's value), w_smooth=0.0046 (=87's 0.002 x2.3), NO lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --d-latent 64 --encoder vit_fourier_hybrid --aux-backbone fourier_mlp --mode history --n-history 2 \
  --d-model 92 --pos-encoding linear --attn-window 4 --token-window 16 --pool mean \
  --vit-fourier-fourier-hidden 270 --vit-fourier-fourier-blocks 2 --vit-fourier-kind ifft \
  --prop-attn-window 32 --prop-fourier-ifft \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.0342 --spatial-signed --w-var-floor 0 --w-logdet 0.004 \
  --w-smooth 0.0046 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own fourier_mlp/history2 aux propagator, --amp, k_max=12, 300 epochs (Section 75/81/82's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 88 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
