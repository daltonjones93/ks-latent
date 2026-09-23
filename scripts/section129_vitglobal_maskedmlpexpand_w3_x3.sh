#!/bin/zsh
# User-directed 2026-09-10, "Section 129": "I guess the other thing worth
# checking is to make the VIT global for the encoder and decoder. I think
# I would just make that change, and keep everything else the same from
# 128. call this 129. queue it after stage 1 and 2 finishes for 128 and
# diagnostics are run."
#
# EXACTLY Section 128 (scripts/section128_vit98_maskedmlpexpand_w3_x3.sh)
# with ONE change: --attn-window 4 (Section 98's windowed-attention
# encoder) is DROPPED entirely, so the encoder/decoder ViT's own attention
# defaults to None -- fully dense/global attention among all tokens,
# instead of Section 98/128's +-4 windowed attention. This isolates
# whether the ENCODER's own locality (as opposed to the propagator's,
# which stays masked_mlp_expand/attn_window=3/expand_factor=3, UNCHANGED)
# has any bearing on whether the whole system stays chaotic or collapses --
# a genuinely different variable than anything else changed across
# Sections 98/100/128.
#
# --token-window 16 (overlapping-patch TOKENIZATION, independent of the
# attention MASK itself -- see ViTAutoencoderConfig's docstring) is kept
# unchanged, per "keep everything else the same": it still has well-defined
# meaning with global attention (overlapping input patches, but every
# token now attends to every other token, not just nearby ones).
#
# Regularizers, propagator, epochs, curriculum: ALL unchanged from 128
# (Section 98's w_decorr=0/w_var=0.02/w_spatial=0.04 SIGNED/w_var_floor=0/
# w_logdet=0.008/w_smooth=0.003, masked_mlp_expand at --prop-attn-window 3
# --masked-mlp-expand-factor 3, --full-propagator, 200 Stage-1 epochs +
# 300 Stage-2 epochs at the Section 52/98/100 k12 curriculum).
#
# QUEUED (user-directed): this script is launched by
# scripts/queue_section129_after_128.sh, which waits for Section 128's own
# Stage 1 + Stage 2 to finish AND for Gate 3/4 diagnostics to complete on
# that checkpoint (not just Stage 2 training) before starting this run --
# avoids GPU contention and keeps Section 128's own results uncorrupted by
# a second concurrent job on the same machine (same practice as Section 98
# queuing behind Section 97).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section129_vitglobal_dmodel56_tokenmlp_propmaskedmlpexpand_w3_x3_200ep

echo "=== [1/3] Stage 1: Section 98's vit encoder+decoder with attn_window REMOVED (d_model=56, pos_encoding=linear, GLOBAL/dense attention -- no --attn-window flag, token_window=16, pool=token_mlp/dec_pool=token_mlp reduction=8/hidden=128) + masked_mlp_expand propagator (attn_window=3, expand_factor=3, UNCHANGED from 128), Section 98's EXACT regularizers (w_decorr=0, w_var=0.02, w_spatial=0.04 SIGNED, w_var_floor=0, w_logdet=0.008, w_smooth=0.003, NO lambda_z), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone masked_mlp_expand --mode markovian \
  --d-model 56 --pos-encoding linear --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --prop-attn-window 3 --masked-mlp-expand-factor 3 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own masked_mlp_expand aux, --amp, k_max=12, 300 epochs (Section 52/98/100/128's shared curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 129 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
