#!/bin/zsh
# User-directed 2026-09-05, "Section 98": "I think the fourier_mlp
# improves conditioning too. but not sure how much it matters. make 98
# exactly 97 but remove the fourier mlp in both encoder and decoder.
# don't change anything else."
#
# Identical to Section 97 (scripts/section97_75pct_wsmooth_2x95.sh) in
# every respect EXCEPT the encoder: --encoder vit_fourier_hybrid ->
# --encoder vit (plain ViT, no Fourier branch at all). All the
# vit_fourier_hybrid-specific / fourier_mlp-specific flags
# (--vit-fourier-fourier-hidden/--vit-fourier-fourier-blocks/
# --vit-fourier-kind/--vit-fourier-enc-out-modes/
# --fourier-mlp-dec-fno-modes) are dropped entirely -- they don't apply
# to plain --encoder vit. The ViT sub-config itself is UNCHANGED
# (d_model=56, pos_encoding=linear, attn_window=4, token_window=16,
# pool=token_mlp/dec_pool=token_mlp reduction=8/hidden=128) -- "don't
# change anything else" means the ViT branch's own hyperparameters stay
# exactly as they were inside Section 97's hybrid, not rebalanced to
# compensate for the removed Fourier branch.
#
# Verified this turn by direct instantiation: plain ViT alone (this
# config) = 308,157 params, vs. Section 97's full hybrid total of
# 388,091 (= vit 308,101 + fourier 79,990) -- confirms removing the
# Fourier branch just drops the ~80K-param Fourier piece, the ViT branch
# itself is essentially unchanged in isolation (a ~56-param difference,
# noise from construction, not a real architecture change).
#
# Propagator/regularizers UNCHANGED from 97: Section 52's EXACT
# mlp/markovian propagator (hidden=128/n_blocks=3), w_var=0.02,
# w_spatial=0.04 (SIGNED), w_var_floor=0, w_logdet=0.008, w_smooth=0.003,
# NO lambda_z, w_decorr=0.
#
# Context/motivation (user, this turn): "I think the fourier_mlp improves
# conditioning too. but not sure how much it matters" -- direct test of
# the Fourier branch's contribution, following a discussion this session
# noting that EVERY vit_fourier_hybrid checkpoint tested (81, 85, 89-97)
# got D4's "Genuine (approximately) equivariant translation
# representation found" verdict, while Section 52 (plain ViT + mlp/
# markovian propagator, NO Fourier branch) is the one architecture in the
# whole project that consistently got the opposite verdict ("No clean
# linear translation representation found") -- predicted (not yet
# confirmed) that removing the Fourier branch here would likely lose D4
# equivariance, matching Section 52's own precedent, regardless of
# whether raw accuracy/conditioning are helped or hurt.
#
# Queued behind Section 97 (still training on MPS at launch time) to
# avoid GPU contention -- smoke-tested once 97 finished.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section98_vitonly_dmodel56_tokenmlp_propmlpmarkovian52_wspatial04_wsmooth003_200ep

echo "=== [1/3] Stage 1: PLAIN vit encoder+decoder (NO Fourier branch -- d_model=56, pool=token_mlp/dec_pool=token_mlp reduction=8/hidden=128, everything else identical to Section 97's ViT sub-config) + Section 52's EXACT mlp/markovian propagator (hidden=128/n_blocks=3), w_decorr=0, w_var=0.02, w_spatial=0.04 (SIGNED), w_var_floor=0, w_logdet=0.008, w_smooth=0.003, NO lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone mlp --mode markovian \
  --d-model 56 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
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

echo "=== Section 98 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
