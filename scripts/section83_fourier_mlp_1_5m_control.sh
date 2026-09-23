#!/bin/zsh
# User-directed 2026-09-04, "Section 83": controlled-comparison run
# following "we don't really know if the hybrid is the improvement, the
# models did get larger" -- Section 81's AE (ViT+FourierIFFTBody hybrid,
# 1,509,438 params) beat Section 75's AE (masked fourier_mlp, 1,002,332
# params) at best_val_kmax_mse (0.0286 vs 0.0392), but the two AEs differ
# in BOTH architecture AND size simultaneously, confounding the
# comparison. User chose: "grow 75 to match 81's ~1.5M params" -- scale
# UP Section 75's ORIGINAL fourier_mlp architecture (same masking,
# attn_window=8/dec_attn_window=22, fourier_ifft_readout, same
# propagator, same regularizers) to match Section 81's AE size exactly,
# then compare the two ~1.5M-param results directly. If Section 75's
# architecture at 1.5M still trails Section 81's 0.0286, that's real
# evidence for the hybrid architecture itself, not just capacity; if it
# closes the gap or matches, capacity was the actual explanation.
#
# ONLY the AE's hidden width changes from Section 75's original run
# (224->282) to hit the size target -- n_blocks, attn_window,
# dec_attn_window, fourier_ifft_readout, propagator, and every
# regularizer weight are all UNCHANGED from Section 75, isolating size as
# the only additional variable relative to the original Section 75 run.
#
# Sizing verified by direct instantiation sweep this turn: hidden=282,
# n_blocks=4 (attn_window=8, dec_attn_window=22, fourier_ifft_readout=True)
# -> 1,509,368 params, essentially an EXACT match to Section 81's
# 1,509,438 (ratio 1.0000). Propagator unchanged from Section 75/81
# (hidden=480/n_blocks=2, attn_window=22, fourier_ifft_readout) ->
# 1,005,046 params.
#
# Verified this turn before launch: a real 4-epoch Stage 1 smoke run
# (val_recon_final=0.014574, confirmed 1,509,368 AE params via checkpoint
# inspection) and a 4-epoch Stage 2 warm-start smoke run -- val_kmax_mse
# only reached 0.188 by epoch 3 (k_now=6), notably WORSE than both
# Section 75's own smoke trajectory at this size (~1M) and Section 81's
# hybrid smoke trajectory, an early (not yet conclusive at just 4 epochs)
# signal that scaling up this architecture's raw capacity alone does NOT
# help and may hurt -- consistent with Sections 77/78's earlier finding
# that more capacity within the SAME masked-fourier_mlp family doesn't
# straightforwardly improve fit. No code changed this turn (pure
# hyperparameter change: --fourier-mlp-hidden 224->282), so the existing
# test suite is unaffected -- not re-run. Smoke artifacts cleaned up
# before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section83_fouriermlp_1_5m_hidden282_section75regs_200ep

echo "=== [1/3] Stage 1: masked fourier_mlp encoder/decoder (Section 75's EXACT architecture -- attn_window=8, dec_attn_window=22, fourier_ifft_readout -- hidden SCALED UP 224->282 to match Section 81's ~1.5M AE size) + Section 75's exact fourier_mlp propagator (attn_window=22, fourier_ifft_readout, hidden=480/n_blocks=2), w_decorr=0, w_var=0.01, w_spatial=0.01 (SIGNED), w_var_floor=0, w_logdet=0.008, lambda_z=0.0002 (from epoch 0), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder fourier_mlp --aux-backbone fourier_mlp --mode history --n-history 2 \
  --attn-window 8 --prop-attn-window 22 --dec-attn-window 22 --fourier-ifft-readout --prop-fourier-ifft \
  --fourier-mlp-hidden 282 --fourier-mlp-blocks 4 \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.01 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --lambda-z 0.0002 --reg-start-epoch 0 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own fourier_mlp/history2 aux propagator, --amp, k_max=12, 300 epochs (Section 75/81's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 83 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
