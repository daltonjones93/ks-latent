#!/bin/zsh
# User-directed 2026-09-06, "Section 100": "I would really like to enforce
# local dynamics architecturally. Here's my thought. Let's have the
# encoder be a ViT like section 98 with a single layer and limited
# attention window (add more parameters to compensate for fewer layers).
# Let's include a token mlp in that encoder that respects the local
# attention window. For the propagator let's use a single layer mlp with
# more parameters with limited attention window (say like 4). Please add
# the proper number of parameters to match the size of the propagator in
# 98, off diagonal terms (from the attention window) shouldn't count. For
# the decoder, let's also try to enforce this local structure from the
# encoder (but also just stick with one layer.) Make this change and try
# this as section 100. Drop the regularizer params to the levels from
# section 93."
#
# Direct test of the OPPOSITE of the fit_latent_pde.py finding this same
# turn (SINDy-style local sparse regression on Section 98's real latent
# trajectories collapsed to a fixed point, R^2~=0.005, D_KY=0): instead of
# abandoning locality, this ARCHITECTURALLY enforces a local, windowed
# propagator -- with real learned nonlinearity (unlike SINDy's 8-term
# linear/quadratic library) -- to test whether H-PROP's repeated finding
# (a propagator recovers chaos IFF it has a GLOBAL receptive field;
# local-reach maps always collapse to a fixed point regardless of
# mechanism -- see docs/model-research-summary-9-5-26.md) holds even with
# a much richer local nonlinearity, or whether it was an artifact of every
# prior local backbone (local_mlp, masked_mlp, node) being comparatively
# narrow.
#
# TWO NEW CAPABILITIES built this turn to make this possible:
#
#   1. `--aux-backbone masked_mlp_wide` (new PropagatorConfig backbone,
#      ks_latent/models/propagator.py's _MaskedMLPWideDeltaBody): a SINGLE
#      hidden layer -- MaskedLinearRect(d_latent, hidden, attn_window) ->
#      GELU -> MaskedLinearRect(hidden, d_latent, attn_window) -- unlike
#      "masked_mlp" (dimension-preserving throughout, no separate hidden
#      width). Sizing: attn_window=4 at d_latent=44 only activates ~18%
#      of each MaskedLinearRect's entries (a +-4-token band out of a
#      44-ring), so matching Section 98's mlp/markovian propagator's
#      111,532 total params ACTIVE-for-active ("off diagonal terms ...
#      shouldn't count") requires hidden=6558 (~51x Section 98's own
#      hidden=128) -- verified by exact search over
#      _circular_band_mask_rect's nonzero count (best match: 111,534 vs
#      target 111,532). This unusually wide single hidden layer is a
#      direct, deliberate consequence of the user's own "off diagonal
#      terms shouldn't count" instruction at this narrow a window, not a
#      sizing mistake -- flagged explicitly (see PropagatorConfig's
#      docstring for the full derivation).
#
#   2. `--pool local_token_mlp` / `--dec-pool local_token_mlp` (new
#      ViTAutoencoderConfig pool mode, ks_latent/models/autoencoder_vit.py):
#      a WINDOWED variant of the existing "token_mlp" pool -- the same
#      per-token FFN (d_model -> compressed_dim = d_model//
#      token_mlp_reduction) feeds a LEARNED, circular-band-masked map
#      (MaskedLinearRect, window=attn_window in n_tokens-ring units -- the
#      SAME window already restricting this encoder's own attention, so
#      the pooling step "respects" it exactly) straight from the n_tokens
#      axis to d_latent, instead of token_mlp's fully dense flatten+MLP.
#      Structurally mirrors the existing pool="banded" mode with the
#      compressed per-token representation substituted for the raw
#      d_model one.
#
# Encoder/decoder sizing: Section 98's plain-ViT AE (n_blocks=3, d_model=56,
# pool=token_mlp) totals 308,157 params. With n_blocks=1 (single layer, per
# the user's request) and pool/dec_pool=local_token_mlp, d_model=104 gives
# the closest achievable total (297,527, -3.4%) among d_model values
# divisible by n_heads=4 -- verified by direct search (d_model=100 ->
# 275,595 [-10.5%]; d_model=108 -> 320,087 [+3.9%]); 104 is closest.
# attn_window=4/token_window=16/pos_encoding=linear/patch_size=8 are
# otherwise UNCHANGED from Section 98's ViT sub-config.
#
# Regularizers: dropped to Section 93's levels, per the user's explicit
# request ("Drop the regularizer params to the levels from section 93"):
# w_decorr=0, w_var=0.02, w_spatial=0.019 (SIGNED), w_var_floor=0,
# w_logdet=0.008, w_smooth=0.0005, NO lambda_z.
#
# EXPECTED RESULT (stated before launch, per this project's standing
# practice): given H-PROP's repeated finding across local_mlp/masked_mlp/
# node (every local-receptive-field propagator tried so far collapsed to
# D_KY=0 regardless of mechanism -- softmax, conv, or per-position masked
# weights), the most likely outcome here is ANOTHER collapse (D_KY~0,
# possibly a severe conditioning blowup like Section 99's cond#=9.6e9, or
# a milder fixed-point contraction like the SINDy fit). A genuinely
# positive result (D_KY staying meaningfully above 0, comparable rollout
# accuracy to Section 98) would be the first evidence that H-PROP's
# pattern was about NONLINEARITY RICHNESS (a plain circular conv/masked
# linear being too narrow a function class), not receptive-field width
# itself -- since this backbone's ~51x-wider single hidden layer is a
# substantially richer nonlinear local map than anything tried under that
# finding before. Both outcomes are informative; this is a real,
# uncertain test, not a formality.
#
# Verified this turn: direct instantiation confirms the propagator's
# active (non-zero-masked) param count is 111,534 (target 111,532) and it
# is exactly identity-at-init (zero_init=True); the AE totals 297,527
# params (target 308,157). A real 4-epoch Stage 1 smoke run (--profile
# full, val_recon_final=0.105693, no errors) and a 4-epoch Stage 2
# warm-start smoke run (val_kmax_mse trending down 0.852->0.762 by epoch
# 3) both completed cleanly; smoke artifacts cleaned up before this real
# launch. 24 new unit tests added (masked_mlp_wide + local_token_mlp,
# construction/shape/identity-at-init/gradient-masking/config-validation),
# full suite at 504 passed (up from 480).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section100_vitonly_dmodel104_nblocks1_localtokenmlp_propmaskedmlpwide_h6558_w4_wspatial019_wsmooth0005_200ep

echo "=== [1/3] Stage 1: PLAIN vit encoder+decoder (n_blocks=1, d_model=104, pool=local_token_mlp/dec_pool=local_token_mlp reduction=8, attn_window=4/pos_encoding=linear/token_window=16) + masked_mlp_wide propagator (hidden=6558/attn_window=4, single hidden layer, sized to match Section 98's mlp/markovian propagator's 111,532 ACTIVE params), w_decorr=0, w_var=0.02, w_spatial=0.019 (SIGNED, Section 93's level), w_var_floor=0, w_logdet=0.008, w_smooth=0.0005 (Section 93's level), NO lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone masked_mlp_wide --mode markovian \
  --d-model 104 --vit-n-blocks 1 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool local_token_mlp --dec-pool local_token_mlp --token-mlp-reduction 8 \
  --aux-hidden 6558 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own masked_mlp_wide aux, --amp, k_max=12, 300 epochs (Section 52's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 100 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
