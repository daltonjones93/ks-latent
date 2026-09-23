#!/bin/zsh
# User-directed 2026-09-05, "Section 94": "can we try a section 94 with
# a model 60% of the size of 93. otherwise keep everything else the
# same."
#
# Same model family as 93 (vit_fourier_hybrid, token_mlp pooling both
# encoder and decoder, reduction=8/hidden=128, fourier_kind=ifft,
# enc_out_modes=8, dec_fno_modes=12, attn_window=4/pos_encoding=linear/
# token_window=16), scaled down further:
#   - d_model: 84 -> 64
#   - fourier_hidden: 130 -> 95 (fourier_blocks stays 2)
# Verified this turn by direct instantiation sweep: vit=388,522 (75.3%),
# fourier=127,404 (24.7%), total=515,926 -- 60.7% of Section 93's 849,539
# (closest achievable match to the requested 60% while keeping the same
# ~75/25 ViT/Fourier ratio established since Section 91).
#
# Propagator: UNCHANGED, Section 52's EXACT mlp/markovian recipe
# (hidden=128/n_blocks=3, default full-propagator sizing).
#
# Regularizers: unchanged from 90-93 (w_var=0.02, w_spatial=0.019 SIGNED,
# w_var_floor=0, w_logdet=0.008, w_smooth=0.0005, NO lambda_z, w_decorr=0).
#
# Context: Sections 91->92->93 each improved on the last while SHRINKING
# (91: ~1.53M/0.0162, 92: ~1.52M+token_mlp/0.0135, 93: ~850K (55%)/0.0118,
# best of the whole 71-93 arc on val_kmax_mse, DA skill, AND smoothness
# simultaneously). This tests whether that trend continues at ~60% of
# 93's already-small size.
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.025284, no errors) and a 4-epoch Stage 2 warm-start smoke run (both
# completed without error). No code changed this turn (pure hyperparameter
# combination of existing capabilities). Smoke artifacts cleaned up before
# this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section94_vitfourieriffthybrid_dmodel64_fourierhidden95_tokenmlp_propmlpmarkovian52_wspatial019_wsmooth0005_200ep

echo "=== [1/3] Stage 1: vit_fourier_hybrid encoder+decoder (75%/25% ViT/Fourier split SCALED to ~61% of Section 93: d_model=64/fourier_hidden=95, ~516K total, pool=token_mlp/dec_pool=token_mlp reduction=8/hidden=128, enc_out_modes=8, dec_fno_modes=12) + Section 52's EXACT mlp/markovian propagator (hidden=128/n_blocks=3), w_decorr=0, w_var=0.02, w_spatial=0.019 (SIGNED), w_var_floor=0, w_logdet=0.008, w_smooth=0.0005, NO lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit_fourier_hybrid --aux-backbone mlp --mode markovian \
  --d-model 64 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --vit-fourier-fourier-hidden 95 --vit-fourier-fourier-blocks 2 --vit-fourier-kind ifft \
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

echo "=== Section 94 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
