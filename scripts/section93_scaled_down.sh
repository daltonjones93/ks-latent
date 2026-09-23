#!/bin/zsh
# User-directed 2026-09-05, "Section 93": "huh, I wonder if a smaller
# encoder and decoder could work better as well. we can make that section
# 93. only include token_mlp in 93 if it seems to improve on 91's
# performance in 92. to be clear, use the same model as 91, just scale it
# down. keep the propagator the same."
#
# Conditional resolved: Section 92 (=91 + token_mlp pooling both sides)
# beat Section 91 (best_val_kmax_mse=0.013474 vs 91's 0.016197, also
# better val_recon_final=0.000091 vs 0.000145) -- so per the user's
# explicit condition, Section 93 uses token_mlp pooling (not 91's
# original local/mean split).
#
# Architecture: SAME model family as 91 (vit_fourier_hybrid, 75%/25%
# ViT/Fourier split, fourier_kind=ifft, enc_out_modes=8, dec_fno_modes=12,
# attn_window=4/pos_encoding=linear/token_window=16), SCALED DOWN:
#   - d_model: 116 -> 84
#   - fourier_hidden: 185 -> 130 (fourier_blocks stays 2)
# Verified this turn by direct instantiation sweep: vit=638,893 (75.2%),
# fourier=210,646 (24.8%), total=849,539 -- ~55% of Section 91/92's own
# ~1.52-1.53M, same 75/25 ratio preserved, isolating SIZE as the only
# variable versus 92.
#
# Pooling: token_mlp on BOTH sides (per the resolved conditional above),
# reduction=8/hidden=128 unchanged from 92.
#
# Propagator: UNCHANGED, Section 52's EXACT mlp/markovian recipe
# (hidden=128/n_blocks=3, default full-propagator sizing) -- "keep the
# propagator the same."
#
# Regularizers: unchanged from 90/91/92 (w_var=0.02, w_spatial=0.019
# SIGNED, w_var_floor=0, w_logdet=0.008, w_smooth=0.0005, NO lambda_z,
# w_decorr=0).
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.023697, no errors) and a 4-epoch Stage 2 warm-start smoke run (both
# stages completed without error; early numbers noisy/not predictive at
# just 4 epochs, as usual). No code changed this turn (pure hyperparameter
# combination of existing capabilities). Smoke artifacts cleaned up
# before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section93_vitfourieriffthybrid_dmodel84_fourierhidden130_tokenmlp_propmlpmarkovian52_wspatial019_wsmooth0005_200ep

echo "=== [1/3] Stage 1: vit_fourier_hybrid encoder+decoder (75%/25% ViT/Fourier split SCALED DOWN: d_model=84/fourier_hidden=130, ~850K total vs 91/92's ~1.5M, pool=token_mlp/dec_pool=token_mlp reduction=8/hidden=128, enc_out_modes=8, dec_fno_modes=12) + Section 52's EXACT mlp/markovian propagator (hidden=128/n_blocks=3), w_decorr=0, w_var=0.02, w_spatial=0.019 (SIGNED), w_var_floor=0, w_logdet=0.008, w_smooth=0.0005, NO lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit_fourier_hybrid --aux-backbone mlp --mode markovian \
  --d-model 84 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --vit-fourier-fourier-hidden 130 --vit-fourier-fourier-blocks 2 --vit-fourier-kind ifft \
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

echo "=== Section 93 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
