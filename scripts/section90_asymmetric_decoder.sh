#!/bin/zsh
# User-directed 2026-09-05, "Section 90": "same encoder, decoder and
# propagator [as 89]. except for the decoder, I would like to only apply
# the fft on lower frequency modes ... in the decoder mlp only take in a
# lower dimensional spectrum. however in the decoder I would also like to
# not restrict the output spectrum of the irfft. This way we incentivize
# (hopefully) the embedding not to have as many higher order frequencies.
# maybe have the input modes = 12 to the decoder. Also just for the
# decoder, use global mean pooling for the vit. Finally please set
# w_spatial = .019, and w_smooth = .0005."
#
# Base: Section 89 (scripts/section89_vit85_truncated_local.sh) --
# vit_fourier_hybrid AE (ViT d_model=92, pos_encoding=linear,
# attn_window=4, token_window=16, pool=local/pool_window=8 + FourierIFFTBody
# fourier_hidden=270/n_blocks=2, fourier_kind=ifft, ENCODER out_modes=8) +
# fourier_mlp/history2 propagator (attn_window=44, fourier_ifft_readout,
# hidden=480/n_blocks=2). UNCHANGED from 89 except, ALL on the DECODER side:
#
#   1. --fourier-mlp-dec-fno-modes 12: the decoder's Fourier branch INPUT
#      is restricted to the first 12 rfft modes of z (out of the full 23
#      available at d_latent=44) -- this ALREADY EXISTED as
#      ViTFourierHybridAutoencoderConfig.dec_fno_modes (wired to
#      KSAutoencoderViTFourierHybrid.dec_modes), no new code needed.
#      Genuinely distinct from Section 89's ENCODER-side enc_out_modes=8
#      (which truncates OUTPUT modes, i.e. what the encoder's Fourier
#      branch is allowed to PREDICT before its irfft): this instead
#      truncates INPUT modes -- how much of z's own spectrum the decoder's
#      Fourier branch is allowed to SEE in the first place.
#   2. Decoder's OUTPUT spectrum (dec_out_modes) deliberately left at its
#      default None (full NX//2+1 spectrum) -- "not restrict the output
#      spectrum of the irfft" -- unchanged from 89, which already left this
#      untouched.
#   3. --dec-pool mean: CODE ADDED this turn -- ViTAutoencoderConfig.dec_pool
#      (None default = mirror --pool, unchanged behavior for every existing
#      recipe). When set, KSAutoencoderViT.decode uses THIS mode for its own
#      expansion/un-pooling step while encode keeps using --pool -- here,
#      encoder stays --pool local --pool-window 8 (89's hard-local
#      constraint, intended to organize/smooth the latent) while the
#      DECODER becomes fully global/dense (mean pool's
#      Linear(d_latent, n_tokens*d_model)), so decode isn't ALSO
#      bottlenecked by the same locality constraint deliberately imposed on
#      the encoder. 7 new unit tests in tests/unit/test_autoencoder_vit.py
#      (asymmetric shape/gradient-flow checks, validation of dec_pool's
#      pool_window/bandwidth compatibility in BOTH directions, default-None
#      regression guard), full suite re-run clean (480 passed).
#
# Everything else identical to 89: w_var=0.02, w_var_floor=0, w_logdet=0.008,
# NO lambda_z, w_decorr=0, --amp, --full-propagator, Stage-2 warm-started
# k_max=12 curriculum, 200ep Stage 1 / 300ep Stage 2. Regularizer changes
# requested this turn:
#   4. w_spatial (signed): 0.014 -> 0.019.
#   5. w_smooth: 0.0009 -> 0.0005 (DECREASE this time, not another push up).
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.033949 -- notably BETTER than Section 89's own 4-epoch smoke of
# 0.056751, consistent with the global mean-pooled decoder being more
# expressive than 89's local-pooled one) and a 4-epoch Stage 2 warm-start
# smoke run (val_kmax_mse trending down 0.235->0.120 by epoch 3) under this
# exact flag combination. Smoke artifacts cleaned up before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section90_vitfourieriffthybrid_encoutmodes8_decfnomodes12_poollocalw8_decpoolmean_wspatial019_wsmooth0005_200ep

echo "=== [1/3] Stage 1: vit_fourier_hybrid encoder (ViT local pooling pool_window=8 + FourierIFFTBody enc_out_modes=8) / DECODER (ViT global mean pooling + FourierIFFTBody dec_fno_modes=12 input truncation, FULL output spectrum) + fourier_mlp/history2 propagator (attn_window=44, fourier_ifft_readout, hidden=480/n_blocks=2), w_decorr=0, w_var=0.02, w_spatial=0.019 (SIGNED), w_var_floor=0, w_logdet=0.008, w_smooth=0.0005, NO lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit_fourier_hybrid --aux-backbone fourier_mlp --mode history --n-history 2 \
  --d-model 92 --pos-encoding linear --attn-window 4 --token-window 16 --pool local --pool-window 8 --dec-pool mean \
  --vit-fourier-fourier-hidden 270 --vit-fourier-fourier-blocks 2 --vit-fourier-kind ifft \
  --vit-fourier-enc-out-modes 8 \
  --fourier-mlp-dec-fno-modes 12 \
  --prop-attn-window 44 --prop-fourier-ifft \
  --aux-hidden 480 --aux-blocks 2 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own fourier_mlp/history2 aux propagator, --amp, k_max=12, 300 epochs (Section 75/81/85/89's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 90 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
