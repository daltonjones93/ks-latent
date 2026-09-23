#!/bin/zsh
# User-directed 2026-09-05, "Section 92": "I think we should try 91
# again exactly but let's just add the token_mlp for both encoder and
# decoder."
#
# Identical to Section 91 (scripts/section91_vit75pct_prop52.sh) in every
# respect EXCEPT pooling mechanism: --pool local --pool-window 8 --dec-pool
# mean -> --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8
# --token-mlp-hidden 128 (the new capability built this session -- a
# per-token FFN compresses d_model -> d_model//8 BEFORE flattening, then a
# 2-layer MLP maps the flattened vector to/from d_latent, symmetric on
# both encoder and decoder; see ks_latent/models/autoencoder_vit.py's
# enc_token_ffn/enc_flatten_mlp/dec_flatten_mlp/dec_token_ffn).
#
# Verified this turn by direct instantiation: swapping the pooling
# mechanism barely changes total size or the 75/25 split -- vit=1,138,476
# (74.9%), fourier=381,004 (25.1%), total=1,519,480 (Section 91's own:
# vit=1,149,581 (75.1%), fourier=381,004 (24.9%), total=1,530,585) -- so
# this isolates the POOLING MECHANISM as the only real variable versus 91,
# not size/ratio.
#
# Everything else unchanged from 91: d_model=116/fourier_hidden=185/
# fourier_blocks=2 (75%/25% ViT/Fourier split), fourier_kind=ifft,
# enc_out_modes=8 (encoder Fourier output truncation), dec_fno_modes=12
# (decoder Fourier input truncation), attn_window=4/pos_encoding=linear/
# token_window=16, Section 52's EXACT mlp/markovian propagator (hidden=128/
# n_blocks=3, default full-propagator sizing), w_var=0.02/w_spatial=0.019
# (SIGNED)/w_var_floor=0/w_logdet=0.008/w_smooth=0.0005/NO lambda_z/
# w_decorr=0, --amp, 200ep Stage 1/300ep Stage 2 (Section 52's curriculum).
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.017317 -- notably BETTER than Section 91's own 4-epoch smoke of
# 0.035034, consistent with token_mlp preserving more per-token information
# than local/mean pooling) and a 4-epoch Stage 2 warm-start smoke run (both
# completed without error -- early numbers noisy/not directly comparable,
# likely confounded by running concurrently with Section 91's own
# real training on the same MPS device). No code changed this turn (pure
# hyperparameter/pooling-mode combination of existing capabilities).
# Smoke artifacts cleaned up before this real launch.
#
# Queued behind Section 91 (outer PID chain from
# scripts/section91_vit75pct_prop52.sh) to avoid MPS training contention.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section92_vitfourieriffthybrid_dmodel116_fourierhidden185_tokenmlp_propmlpmarkovian52_wspatial019_wsmooth0005_200ep

echo "=== [1/3] Stage 1: vit_fourier_hybrid encoder+decoder (75%/25% ViT/Fourier split, d_model=116/fourier_hidden=185, pool=token_mlp/dec_pool=token_mlp reduction=8/hidden=128, enc_out_modes=8, dec_fno_modes=12) + Section 52's EXACT mlp/markovian propagator (hidden=128/n_blocks=3), w_decorr=0, w_var=0.02, w_spatial=0.019 (SIGNED), w_var_floor=0, w_logdet=0.008, w_smooth=0.0005, NO lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit_fourier_hybrid --aux-backbone mlp --mode markovian \
  --d-model 116 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
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

echo "=== Section 92 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
