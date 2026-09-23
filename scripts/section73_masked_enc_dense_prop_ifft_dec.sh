#!/bin/zsh
# User-directed 2026-09-03, "Section 73": "let's try an encoder that is
# the same fourier_mlp with attn_window = 4, a dense fourier_mlp
# propagator and a dense inverse fourier mlp for the decoder where the
# inverse fourier mlp applied the ifft not the fft. write the necessary
# code and start the test. keep training parameters and settings the
# same."
#
# New capability: --dec-use-ifft (train_stage1_patched.py CLI flag, wires
# FourierMLPAutoencoderConfig.dec_use_ifft). Forces the decoder to a
# fully-dense structure regardless of --attn-window (which then applies
# to the encoder only) -- the first asymmetric encoder-masked/decoder-
# dense construction in this codebase. The decoder's auxiliary feature
# changes from _fourier_features(z, dec_modes) (forward rfft of the
# latent -- an odd fit for a decode step, kept only for backward
# compatibility) to _inverse_fourier_features(z, NX): treats the raw
# latent z itself as the non-negative-frequency half of a Hermitian
# spectrum (imaginary part zero) and applies irfft to reconstruct an
# NX-length physical-domain signal directly, summed additively with raw z
# into a single dense MLPDeltaBody (never masked -- there's no circular-
# band structure to mask once the ifft output already mixes every latent
# coordinate by construction).
#
# Kept additive with raw z, not replacing it -- same safety reasoning as
# _MaskedRectPath's raw-value path elsewhere in this class: a pure-
# frequency-domain-only encoder/decoder (spectral_mlp, tried and fully
# reverted 2026-09-03) diverged catastrophically in Stage 2
# (val_kmax_mse ~170-190).
#
# Architecture: encoder=fourier_mlp masked (attn_window=4, same as
# Section 71's encoder), propagator=fourier_mlp fully dense (--prop-dense,
# same as Section 72's propagator), decoder=fourier_mlp dense-ifft (NEW,
# --dec-use-ifft). "keep training parameters and settings the same":
# identical hidden/n_blocks sizing (AE hidden=224/n_blocks=4, propagator
# hidden=480/n_blocks=2) and Section 66's regularizers with lambda_z=0.002
# (2x Section 66), active from epoch 0, as Sections 71/72.
#
# Param counts verified this turn by direct instantiation/checkpoint
# inspection: AE (encoder masked attn_window=4, decoder dense-ifft) ->
# 1,030,968 params (vs. Section 71/72's 1,001,432 -- the ifft decoder's
# input_proj is d_latent+NX=300 wide instead of d_latent+2*dec_modes,
# hence the small increase). Propagator (dense, hidden=480/n_blocks=2) ->
# 1,034,444 params, same as Section 72's.
#
# Verified this turn before launch: 8 new unit tests (now 31 total in
# test_autoencoder_fourier_mlp.py) covering _inverse_fourier_features
# directly, the asymmetric masked-encoder/dense-decoder construction,
# input-dim/shape checks, gradient flow, bfloat16 autocast, and checkpoint
# round-trip; full unit+integration suite (see /tmp/full_test_run16.log);
# a real 2-epoch Stage 1 smoke run (val_recon_final=0.032626) confirming
# attn_window=4/dec_use_ifft=True on the saved AE config and
# attn_window=None on the saved propagator config; a Stage 2 warm-start
# smoke run (val_kmax_mse 0.050->0.027 over 2 epochs). Smoke artifacts
# cleaned up before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section73_maskedenc4_denseprop_ifftdec_fouriermlp_section66regs_lambdaz0002_200ep

echo "=== [1/3] Stage 1: masked fourier_mlp encoder (attn_window=4) + dense inverse-fourier-mlp decoder (--dec-use-ifft) + FULLY DENSE fourier_mlp/history2 aux propagator (--prop-dense, hidden=480/n_blocks=2, full-propagator sizing), w_decorr=0, w_var=0.015, w_spatial=0.035 (SIGNED), w_var_floor=0, w_logdet=0.005, lambda_z=0.002 (from epoch 0, 2x Section 66), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder fourier_mlp --aux-backbone fourier_mlp --mode history --n-history 2 \
  --attn-window 4 --prop-dense --dec-use-ifft \
  --fourier-mlp-hidden 224 --fourier-mlp-blocks 4 \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.015 --w-spatial 0.035 --spatial-signed --w-var-floor 0 --w-logdet 0.005 \
  --lambda-z 0.002 --reg-start-epoch 0 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own dense fourier_mlp/history2 aux propagator, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 73 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
