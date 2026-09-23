#!/bin/zsh
# User-directed 2026-09-03, "Section 71": "I want the next experiment to
# follow 66, but use masked fourier_mlp with 1 million params for encoder
# decoder and propagator. increase lambda_z 2x"
#
# New capability used for the first time in a real launch: the masked
# option for the fourier_mlp AUTOENCODER (FourierMLPAutoencoderConfig.
# attn_window), extended this turn from the propagator's existing masked
# fourier_mlp option to the encoder/decoder ("use all the fourier
# coefficients, but only let real variables interact with their
# neighbors" -- originally said about the propagator, now applied to the
# AE too). Both AE and propagator use attn_window=4 (this session's most
# common default; the user did not specify a window size this time).
#
# Architecture: encoder=fourier_mlp (masked raw-value path via
# _MaskedRectPath + dense Fourier-coefficient path, SUMMED), aux/full
# propagator=fourier_mlp (masked raw-value path + dense Fourier path,
# mode=history, n_history=2), attn_window=4 for both. Sized to ~1M params
# each via direct instantiation sweep (verified this turn):
#   AE:         hidden=224, n_blocks=4 -> 1,001,432 params
#   propagator: hidden=480, n_blocks=2 -> 1,004,084 params
# (Section 66's fourier_mlp AE was unmasked and roughly Section-52-scale,
# 200-800K; this is a substantially larger, masked variant of the same
# encoder family.)
#
# Regularizers ("follow 66 ... increase lambda_z 2x"): Section 66's exact
# recipe -- w_decorr=0, w_var=0.015, w_spatial(signed)=0.035, w_var_floor=0,
# w_logdet=0.005 -- with lambda_z doubled from 0.001 to 0.002, active from
# epoch 0 (--reg-start-epoch 0, matching Section 66).
#
# Verified this turn before launch: 30 new/existing unit tests pass
# (test_autoencoder_fourier_mlp.py + test_propagator_fourier_mlp.py), full
# suite 409 passed (no regressions), direct weight-zero-outside-band checks
# for both AE masked crossings, a real 2-epoch Stage 1 smoke run
# (val_recon_final=0.022831) and Stage 2 warm-start smoke run
# (val_kmax_mse 0.048->0.028 over 2 epochs) at this exact architecture/param
# count. Smoke artifacts cleaned up before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section71_masked_fouriermlp_ae1m_prop1m_attn4_section66regs_lambdaz0002_200ep

echo "=== [1/3] Stage 1: masked fourier_mlp AE (hidden=224, n_blocks=4, attn_window=4, ~1M params) + masked fourier_mlp/history2 aux propagator (hidden=480, n_blocks=2, attn_window=4, ~1M params, full-propagator sizing), w_decorr=0, w_var=0.015, w_spatial=0.035 (SIGNED), w_var_floor=0, w_logdet=0.005, lambda_z=0.002 (from epoch 0, 2x Section 66), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder fourier_mlp --aux-backbone fourier_mlp --mode history --n-history 2 \
  --attn-window 4 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own masked fourier_mlp/history2 aux propagator, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 71 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
