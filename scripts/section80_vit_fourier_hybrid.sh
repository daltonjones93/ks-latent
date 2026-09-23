#!/bin/zsh
# User-directed 2026-09-04, "Section 80": "I don't think the spectral
# norm is worth pursuing based on what we're seeing. I want to try the
# following hybrid network for the encoder and decoder I want a Vit for
# function space, the same structure as section 52 plus I want a Fourier
# mlp that only acts on frequency space. Moreover for this Fourier mlp,
# let's make the absolute value of each output of each layer sum to the
# same value which is the original sum of the absolute values of the
# frequency coefficients (ie |c1| +|c2| + ... etc). This way we can
# impose a conservation law, but we don't need to use something as
# complicated as the spectral norm. Then just add the output of the ViT
# and the Fourier mlp I described. Make sure the number of parameters in
# the encoder and decoder and propagator roughly match the size of 75.
# Also please make the ViT and Fourier mlp commensurate in size. Also
# please just use the Fourier mlp from 75 for the propagator. Let's try
# this with the settings from section 75."
#
# New capability: encoder_kind="vit_fourier_hybrid"
# (KSAutoencoderViTFourierHybrid/ViTFourierHybridAutoencoderConfig,
# ks_latent/models/autoencoder_vit_fourier_hybrid.py) -- two parallel
# sub-networks, SUMMED:
#   - vit: an ordinary KSAutoencoderViT, Section 52's exact structure
#     (patch_size=8, n_heads=4, n_blocks=3, mlp_ratio=4, pool="mean",
#     pos_encoding="linear", attn_window=4, token_window=16,
#     readout="linear"), d_model re-tuned to 72 for the size target below.
#   - fourier_encoder/fourier_decoder: ConservedFourierMLP, operating
#     PURELY on Fourier coefficients (no raw-value/masked path at all) --
#     "a Fourier mlp that only acts on frequency space." Every layer's
#     raw output is rescaled (a single per-sample scalar) so its L1 norm
#     exactly equals the L1 norm of the ORIGINAL input Fourier features
#     (computed once) -- an L1 "conservation law," much simpler than
#     spectral normalization: no constraint on the weight matrix itself
#     (no lost capacity from bounding the operator norm), just a
#     post-hoc rescale. Verified directly this turn: the FINAL output's
#     L1 exactly matches the input's L1 (unit tests), AND critically a
#     real 40-epoch Stage-1 smoke run converges cleanly (loss
#     52.6->0.23->0.084->0.046->0.038, val_recon_final=0.012) -- unlike
#     the spectral-norm/"nonexpansive" fourier_mlp variant tried earlier
#     today, which plateaued at val_recon~0.36 even after 40 epochs
#     (crippled capacity from the per-layer operator-norm bound) and was
#     abandoned. This L1-conservation approach does NOT have that
#     capacity problem, since it never constrains the weights themselves.
#
# Propagator: "just use the fourier mlp from 75 for the propagator" --
# Section 75's EXACT propagator, unchanged (masked fourier_mlp,
# attn_window=22 -- effectively fully dense at d_latent=44 -- +
# fourier_ifft_readout, hidden=480/n_blocks=2, mode=history, n_history=2).
#
# Sizing ("roughly match the size of 75 ... make the ViT and Fourier mlp
# commensurate"): verified by direct instantiation sweep this turn --
# ViT (d_model=72, Section 52's other hyperparameters unchanged) ->
# 487,854 params. ConservedFourierMLP (hidden=246, n_blocks=3, encoder+
# decoder combined) -> 513,948 params. Total AE = 1,001,802 (99.9% of
# Section 75's 1,002,332), with the two branches commensurate (ratio
# 0.95). Propagator = 1,005,046 (identical to Section 75's, reused
# unchanged).
#
# Regularizers/curriculum: "try this with the settings from section 75"
# -- Section 75's exact recipe (w_var=0.01, w_spatial=0.01 signed,
# w_logdet=0.008, lambda_z=0.0002, active from epoch 0) and Stage-2
# curriculum (k_max=12, 300 epochs, k_warmup_epochs=210/k_mid=8/
# k_mid_epochs=175 -- NOT the extended k_max=20/500-epoch curriculum used
# for Sections 76-79, since the user said "the settings from section 75"
# specifically).
#
# Verified this turn before launch: 12 new unit tests (construction,
# shapes, gradient flow, checkpoint round-trip, bfloat16, the L1-
# conservation property itself at every layer not just the final output,
# config d_latent/NX properties -- a real gap found and fixed: generic
# code (train_stage1's regularizer setup, every Gate 3/4 script) reads
# ae_cfg.d_latent/ae_cfg.NX directly, which this nested-vit-subconfig
# class needed explicit @property passthroughs for); full unit+
# integration suite (see /tmp/full_test_run24.log); the 40-epoch Stage 1
# smoke run above; a 4-epoch Stage 2 warm-start smoke run confirming no
# crashes (too few epochs for a real trend, k_max=12 curriculum barely
# started).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section80_vitfourierhybrid_dmodel72_fourierhidden246blocks3_section75regs_200ep

echo "=== [1/3] Stage 1: vit_fourier_hybrid encoder/decoder (ViT d_model=72, Section 52 structure + ConservedFourierMLP hidden=246/n_blocks=3, L1-conserving, summed) + Section 75's exact fourier_mlp propagator (attn_window=22, fourier_ifft_readout, hidden=480/n_blocks=2), w_decorr=0, w_var=0.01, w_spatial=0.01 (SIGNED), w_var_floor=0, w_logdet=0.008, lambda_z=0.0002 (from epoch 0), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit_fourier_hybrid --aux-backbone fourier_mlp --mode history --n-history 2 \
  --d-model 72 --pos-encoding linear --attn-window 4 --token-window 16 --pool mean \
  --vit-fourier-fourier-hidden 246 --vit-fourier-fourier-blocks 3 \
  --prop-attn-window 22 --prop-fourier-ifft \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own fourier_mlp/history2 aux propagator (Section 75's exact architecture), --amp, k_max=12, 300 epochs (Section 75's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 80 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
