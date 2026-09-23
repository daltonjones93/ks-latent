#!/bin/zsh
# User-directed 2026-09-10, "Section 128": "I realize it's a long shot, but
# can we run another experiment. I want to try the encoder and decoder pair
# from 98, along with the regularizers from 98. but I want to use a masked
# mlp with three layers, attention_window=3 for the propagator. the thing I
# would like to do though is in the middle layer of the mlp, expand the
# dimension 3x, but in this expanded dimensions, still respect the
# attention window (I guess it would be 9 in that case, so only a very
# limited number of neighboring values interact.) make sure that the whole
# mlp I've described only lets a small number of neighbors interact."
#
# NEW: `backbone="masked_mlp_expand"` (ks_latent/models/propagator.py's
# `_MaskedMLPExpandDeltaBody`, added this turn) -- three `MaskedLinearRect`
# layers (input_proj: d_latent->hidden, mid_proj: hidden->hidden -- "the
# middle layer" -- output_proj: hidden->d_latent), hidden =
# masked_mlp_expand_factor * d_latent (default 3). All three share
# attn_window referenced against d_latent's own ring (ref_dim=d_latent,
# the SAME convention Section 100's masked_mlp_wide already established).
# Verified directly, not just asserted: this makes the user's own "I guess
# it would be 9" arithmetic exact -- mid_proj's one-sided neighbor radius
# in hidden-index units works out to attn_window*expand_factor=3*3=9
# (test_masked_mlp_expand_mid_proj_radius_matches_window_times_expand_factor,
# checked directly against the built mask at d_latent=44/window=3/factor=3:
# max circular distance connected = 9, row sum = 2*9+1=19 exactly). No
# LayerNorm anywhere (same reasoning as masked_mlp/masked_mlp_wide: it
# would reintroduce full global coupling across the very axis a circular
# mask is trying to respect). zero_init zeros output_proj only, matching
# every other backbone's identity-at-init property. 12 new unit tests
# added (construction/shape/identity-at-init/masked-gradient-zero/radius-
# derivation/dense-when-window-covers-ring/config-validation), full suite
# re-run for regressions.
#
# Encoder/decoder + regularizers: Section 98's EXACT vit config, UNCHANGED
# (scripts/section98_no_fourier_branch.sh) -- plain ViT, no Fourier branch,
# d_model=56, n_blocks=3 (default, not overridden), pos_encoding=linear,
# attn_window=4 (the ENCODER's own window -- kept exactly as Section 98
# had it; NOT the same value as the propagator's attn_window=3 below, see
# --prop-attn-window), token_window=16, pool=token_mlp/dec_pool=token_mlp
# reduction=8/hidden=128, w_decorr=0, w_var=0.02, w_spatial=0.04 (SIGNED),
# w_var_floor=0, w_logdet=0.008, w_smooth=0.003, NO lambda_z. This is a
# genuinely different encoder config than Section 100's own follow-up
# (which used n_blocks=1/d_model=104/local_token_mlp pooling AND Section
# 93's lighter regularizer levels) -- the user asked for 98's pair
# specifically this time, not 100's size-matched variant.
#
# Propagator attn_window vs encoder attn_window: --attn-window (4) sets
# the ENCODER's own window (Section 98's value, unchanged); the
# propagator's window is set INDEPENDENTLY via --prop-attn-window 3 (the
# CLI's existing override for exactly this situation -- see
# --prop-attn-window's own help text, "I want the attn_window to be much
# larger" precedent, used here in the opposite direction: smaller,
# per the user's explicit "attention_window=3 for the propagator").
#
# Verified this turn by direct instantiation before launch: masked_mlp_expand
# at d_latent=44 (Section 98's own d_latent), attn_window=3, expand_factor=3
# builds cleanly (hidden=132), is exactly identity at init, and a real
# --profile smoke Stage 1 run (2 epochs) plus a Stage 2 warm-start smoke run
# (2 epochs) both completed with no errors.
#
# EXPECTED RESULT (stated before launch, per this project's standing
# practice, and the user's own "long shot" framing): H-PROP's repeated
# finding (every local-receptive-field propagator tried so far -- local_mlp,
# masked_mlp, node, masked_mlp_wide/Section 100 -- collapsed to D_KY~0
# regardless of mechanism) predicts the same outcome here. This backbone's
# richer, deeper (3-layer, expanding-then-contracting) local nonlinearity is
# a genuinely different function class than anything tried under that
# finding before (masked_mlp_wide was 1 hidden layer, not 3; local_mlp/node
# use conv/RK4 vector fields, not a masked bottleneck-inverted MLP), so a
# positive result (D_KY staying meaningfully above 0) would be new evidence
# about WHICH local architectures can or can't escape H-PROP's pattern, not
# a formality re-confirming it. Both outcomes are informative.
#
# Phase-1-first (matching Section 100's own practice): Stage 1 (joint,
# --full-propagator) then a latent covariance spectrum + D7/D8 bandedness
# check on the frozen result BEFORE committing to the full Stage 2
# warm-start + Gate 3/4 run -- check via this script's own [2/3] step and
# (once Stage 2 finishes) visualize_rollout.py before trusting a Gate 3
# Lyapunov run's cost.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section128_vitonly_dmodel56_tokenmlp_propmaskedmlpexpand_w3_x3_200ep

echo "=== [1/3] Stage 1: Section 98's EXACT vit encoder+decoder (d_model=56, pos_encoding=linear, attn_window=4, token_window=16, pool=token_mlp/dec_pool=token_mlp reduction=8/hidden=128) + NEW masked_mlp_expand propagator (attn_window=3, expand_factor=3, THREE MaskedLinearRect layers with the middle one at 3x width), Section 98's EXACT regularizers (w_decorr=0, w_var=0.02, w_spatial=0.04 SIGNED, w_var_floor=0, w_logdet=0.008, w_smooth=0.003, NO lambda_z), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone masked_mlp_expand --mode markovian \
  --d-model 56 --pos-encoding linear --attn-window 4 --token-window 16 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own masked_mlp_expand aux, --amp, k_max=12, 300 epochs (Section 52/98/100's shared curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 128 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
