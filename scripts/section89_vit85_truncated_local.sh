#!/bin/zsh
# User-directed 2026-09-05, "Section 89": "rerun 85, but truncate the
# frequencies out of the encoder fourier_mlp to only keep the lowest 22
# frequencies. also can we increase w_smooth to .0009, and w_spatial to
# .014. leave the decoder alone. and set the --prop-attn-window to 44.
# finally can we make the output pooling local for the encoder vit (with
# kernel size 12 or something)." Clarified via questions (exact values
# 22/12 don't map onto valid settings -- see below): enc_out_modes=8,
# pool_window=8.
#
# Base: Section 85 (scripts/section85_vit81_wsmooth.sh) -- vit_fourier_hybrid
# AE (ViT d_model=92, pos_encoding=linear, attn_window=4, token_window=16
# + FourierIFFTBody fourier_hidden=270/n_blocks=2, fourier_kind=ifft) +
# fourier_mlp/history2 propagator (attn_window=22->44, fourier_ifft_readout,
# hidden=480/n_blocks=2). UNCHANGED from 85 except:
#
#   1. CODE ADDED this turn: ViTFourierHybridAutoencoderConfig.enc_out_modes/
#      dec_out_modes (ks_latent/config.py), threaded into
#      KSAutoencoderViTFourierHybrid's FourierIFFTBody construction
#      (autoencoder_vit_fourier_hybrid.py) -- truncates how many frequency
#      coefficients that branch predicts before its irfft (torch.fft.irfft
#      zero-pads any missing high modes automatically, so this is an exact
#      low-pass truncation of that branch's own output). New CLI flags
#      --vit-fourier-enc-out-modes/--vit-fourier-dec-out-modes. 8 new unit
#      tests in tests/unit/test_autoencoder_vit_fourier_hybrid.py, full
#      suite re-run clean (466 passed).
#   2. --vit-fourier-enc-out-modes 8 (decoder left alone, per request --
#      dec_out_modes stays None/full spectrum). NOTE: full spectrum at
#      d_latent=44 is only 44//2+1=23 modes -- the user's literal "22" would
#      have removed just the single highest (Nyquist) mode, essentially no
#      truncation; clarified to 8 (a real low-pass) via question.
#      CAVEAT (documented in config.py, worth repeating here): encode(u) =
#      vit.encode(u) + fourier_encoder(feats) sums an UNCONSTRAINED vit
#      branch with the now band-limited Fourier branch -- this does NOT
#      guarantee the summed z itself is band-limited, only that the
#      Fourier branch's OWN contribution is.
#   3. --pool local --pool-window 8 (was --pool mean). n_tokens=32 (NX=256/
#      patch_size=8) requires pool_window to divide 32 AND d_latent=44 to be
#      divisible by n_sites=n_tokens//pool_window; the user's literal "12"
#      doesn't divide 32 at all (ValueError). Of the two divisors of 32 that
#      also satisfy d_latent's constraint (8 -> 4 sites of 11 channels; 16
#      -> 2 sites of 22 channels), clarified to 8 via question -- finer-
#      grained locality, 4 separate local sites across the domain. This is
#      a HARD architectural locality constraint on the ViT branch's own
#      pooling (unlike every soft regularizer used elsewhere in this arc) --
#      per this project's own repeatedly-confirmed finding (masked_mlp
#      encoder Section 17, banded pooling Sections 24-25), this class of
#      change has consistently cost real reconstruction accuracy every time
#      it's been tried -- confirmed again in this turn's 4-epoch smoke test
#      (val_recon_final=0.056751 vs Section 85's own 4-epoch smoke of
#      0.009089, ~6x worse this early).
#   4. w_spatial (signed): 0.012 -> 0.014.
#   5. w_smooth: 0.00075 -> 0.0009.
#   6. --prop-attn-window: 22 -> 44. NOTE: this is a functional NO-OP --
#      d_latent=44 saturates to fully-dense at window>=22 (d_latent//2), so
#      22 and 44 give the IDENTICAL mask (verified: both all-True). Included
#      only because it was explicitly requested; does not change model
#      behavior at all versus Section 85.
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.056751, no errors -- confirms the expected accuracy cost from #3 above)
# and a 4-epoch Stage 2 warm-start smoke run (val_kmax_mse trending down
# 0.43->0.30 by epoch 3) under this exact flag combination. Smoke artifacts
# cleaned up before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section89_vitfourieriffthybrid_encoutmodes8_poollocalw8_wspatial014_wsmooth0009_200ep

echo "=== [1/3] Stage 1: vit_fourier_hybrid encoder/decoder (ViT d_model=92 with LOCAL pooling pool_window=8 + FourierIFFTBody hidden=270/n_blocks=2, ENCODER Fourier branch truncated to enc_out_modes=8, decoder untouched) + fourier_mlp/history2 propagator (attn_window=44, fourier_ifft_readout, hidden=480/n_blocks=2), w_decorr=0, w_var=0.02, w_spatial=0.014 (SIGNED), w_var_floor=0, w_logdet=0.008, w_smooth=0.0009, NO lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit_fourier_hybrid --aux-backbone fourier_mlp --mode history --n-history 2 \
  --d-model 92 --pos-encoding linear --attn-window 4 --token-window 16 --pool local --pool-window 8 \
  --vit-fourier-fourier-hidden 270 --vit-fourier-fourier-blocks 2 --vit-fourier-kind ifft \
  --vit-fourier-enc-out-modes 8 \
  --prop-attn-window 44 --prop-fourier-ifft \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.014 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.0009 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own fourier_mlp/history2 aux propagator, --amp, k_max=12, 300 epochs (Section 75/81/85's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 89 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
