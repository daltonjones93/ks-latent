#!/bin/zsh
# User-directed 2026-09-03, "Section 70": follow-up to the depth-vs-width
# discussion -- "let's try n_blocks=6, d_model=68. train with the same
# regularizer params as 63 (start lambda_z from the beginning. the
# propagator should be the same across stage 1 and 2 and should be a
# fourier mlp. make spatial signed .02 and lambda_z .003."
#
# Motivation (from the preceding discussion, not repeated here in full):
# attn_window restricts each ViT block to local interactions, so more
# (narrower) blocks directly extends the effective receptive field
# (~n_blocks * attn_window tokens) and adds more rounds of attention's
# non-expansive convex-combination mixing -- the same mechanism already
# implicated in explaining why vit AEs show better long-rollout stability
# than fourier_mlp/spectral_mlp AEs (Sections 65-67 discussion). This
# tests whether trading width for depth, at a similar total param count,
# buys more of that same effect.
#
# New capability added for this run: --vit-n-blocks (train_stage1_patched.py
# CLI flag, wires ViTAutoencoderConfig.n_blocks -- no such override existed
# before; --d-model already existed but always paired with the config's
# default n_blocks=3). Verified directly: d_model=68/n_blocks=6/attn_window=4
# -> 779402 params (ratio 0.955 vs. Section 52's 816342 baseline) --
# checked directly by instantiating, and via a real 2-epoch Stage 1 + 2
# smoke run confirming checkpoint round-trip and Stage 2 warm-start.
#
# Everything else about the ViT unchanged from Section 52/69: attn_window=4
# (Section 69's reverted value, not 6), pos_encoding=linear, token_window=16,
# patch_size=8/pool=mean/n_heads=4/mlp_ratio=4 (defaults), readout=linear
# (default, the readout='mlp' option stays unused here too).
#
# Propagator ("should be the same across stage 1 and 2 and should be a
# fourier mlp"): backbone=fourier_mlp, mode=history, n_history=2,
# hidden=140/n_blocks=2 (Sections 65-69's established Section-52-scale
# sizing, 111344 params) -- same architecture for Stage 1 (aux,
# --full-propagator sizing) and Stage 2 (warm-started via
# --init-prop-checkpoint).
#
# Regularizers ("same regularizer params as 63 ... start lambda_z from
# the beginning ... make spatial signed .02 and lambda_z .003"): Section
# 63's recipe (w_decorr=0, w_var=0.02, w_var_floor=0, w_logdet=0.0035) with
# two explicit overrides -- w_spatial(signed)=0.02 (up from 63's 0.01) and
# lambda_z=0.003 (up from 63's 0.001) -- and --reg-start-epoch 0 (NOT 30
# like Section 63; "start lambda_z from the beginning").
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section70_vit_dmodel68_nblocks6_attn4_fouriermlpprop_wspatialsigned002_lambdaz0003fromepoch0_200ep

echo "=== [1/3] Stage 1: vit AE (d_model=68, n_blocks=6, attn_window=4, pos_encoding=linear, token_window=16) + fourier_mlp/history2 aux propagator (hidden=140/n_blocks=2, full-propagator sizing), w_decorr=0, w_var=0.02, w_spatial=0.02 (SIGNED), w_var_floor=0, w_logdet=0.0035, lambda_z=0.003 (from epoch 0), NO delta_cap, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone fourier_mlp --mode history --n-history 2 \
  --pos-encoding linear --attn-window 4 --token-window 16 --d-model 68 --vit-n-blocks 6 \
  --aux-hidden 140 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.02 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --lambda-z 0.003 --reg-start-epoch 0 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own fourier_mlp/history2 aux propagator, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 70 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
