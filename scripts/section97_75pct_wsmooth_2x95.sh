#!/bin/zsh
# User-directed 2026-09-05, "Section 97": "let's make section 97 to have
# the same settings as 96, with models that are 75% the size, but
# w_smooth twice as large as 95."
#
# Base: Section 96 (scripts/section96_wsmooth0_wspatial04.sh) -- same
# vit_fourier_hybrid architecture family (token_mlp pooling both encoder
# and decoder, reduction=8/hidden=128, fourier_kind=ifft, enc_out_modes=8,
# dec_fno_modes=12, attn_window=4/pos_encoding=linear/token_window=16),
# Section 52's EXACT mlp/markovian propagator (hidden=128/n_blocks=3),
# w_var=0.02/w_var_floor=0/w_logdet=0.008/NO lambda_z/w_decorr=0 unchanged.
# Two changes from 96:
#   1. Model size: d_model 64->56, fourier_hidden 95->70 (fourier_blocks
#      stays 2). Verified this turn by direct instantiation sweep:
#      vit=308,101 (79.4%), fourier=79,990 (20.6%), total=388,091 --
#      75.3% of Section 96's 515,926 (closest achievable match to the
#      requested 75%; the exact 75/25 ViT/Fourier ratio used since
#      Section 91 drifts slightly toward ViT-heavier at this scale --
#      d_model must be divisible by n_heads=4, which constrains the
#      achievable grid).
#   2. w_smooth: 96's 0 (disabled) -> 0.003, EXPLICITLY 2x Section 95's
#      0.0015 (not 2x 96's own value, which was 0) -- w_spatial stays at
#      96's 0.04, unchanged.
#
# Context: Sections 93->94 (size alone) plateaued; 95->96 (w_spatial
# alone, 0.03->0.04, w_smooth removed) gave a clean monotonic
# accuracy/DA-skill tradeoff (accuracy worsens, DA skill improves,
# smoothness worsens, cond# worsens, all monotonically). This tests
# BOTH levers moved at once from 96 -- shrinking size further (which
# plateaued rather than helped at 94's scale) while adding back a
# nonzero w_smooth (which was posited, then ruled out, as compounding
# 95's regression -- 96 without it regressed MORE, not less, at higher
# w_spatial) at a value between 95's 0.0015 and above.
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.029899, no errors) and a 4-epoch Stage 2 warm-start smoke run (both
# completed without error). No code changed this turn (pure hyperparameter
# combination of existing capabilities). Smoke artifacts cleaned up before
# this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section97_vitfourieriffthybrid_dmodel56_fourierhidden70_tokenmlp_propmlpmarkovian52_wspatial04_wsmooth003_200ep

echo "=== [1/3] Stage 1: vit_fourier_hybrid encoder+decoder (75% of Section 96's size: d_model=56/fourier_hidden=70, pool=token_mlp/dec_pool=token_mlp reduction=8/hidden=128, enc_out_modes=8, dec_fno_modes=12) + Section 52's EXACT mlp/markovian propagator (hidden=128/n_blocks=3), w_decorr=0, w_var=0.02, w_spatial=0.04 (SIGNED, =96's value), w_var_floor=0, w_logdet=0.008, w_smooth=0.003 (2x Section 95's 0.0015), NO lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit_fourier_hybrid --aux-backbone mlp --mode markovian \
  --d-model 56 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --vit-fourier-fourier-hidden 70 --vit-fourier-fourier-blocks 2 --vit-fourier-kind ifft \
  --vit-fourier-enc-out-modes 8 \
  --fourier-mlp-dec-fno-modes 12 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.04 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.003 \
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

echo "=== Section 97 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
