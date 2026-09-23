#!/bin/zsh
# User-directed 2026-09-04, "Section 84": "section 52 was the best run
# though right? I would like to start a training ... that uses the vit
# encoder and decoder from 52, uses the fourier mlp for the propagator
# (same for both stage 1 and stage 2) and keeps the regularization
# parameters the same except increases w_spatial_signed slightly by
# 1.35x. make the fourier mlp markovian, and if you can, put more
# parameters in the part of the model dealing with the frequency
# coefficients and less on the part dealing with real function values."
#
# Base: Section 52 (scripts/section52_spatial_signed_tuned5.sh) -- the
# ViT (d_model=96, pos_encoding=linear, attn_window=4, token_window=16)
# encoder/decoder, mode=markovian, w_var=0.02, w_decorr=0, w_var_floor=0,
# w_logdet=0.0035, no lambda_z/delta_cap, --amp, Stage-2 warm-started
# k_max=12 curriculum (k_warmup_epochs=210/k_mid=8/k_mid_epochs=175),
# 200ep Stage 1 / 300ep Stage 2. UNCHANGED from 52 except:
#   1. w_spatial (signed): 0.01 -> 0.0135 (x1.35, per request).
#   2. --aux-backbone mlp -> fourier_mlp (Section 74/75/81's masked-raw +
#      dense/ifft-Fourier propagator body, in place of 52's plain
#      residual-MLP markovian propagator). Used identically for Stage 1
#      (--full-propagator) AND Stage 2 (warm-started from Stage 1's own
#      checkpoint, so no separate Stage-2 propagator flags needed --
#      --init-prop-checkpoint loads the exact saved architecture).
#
# CODE CHANGE required and made this turn: backbone="fourier_mlp" was
# previously mode="history" ONLY (PropagatorConfig.__post_init__ raised
# unconditionally for mode="markovian" -- "a single current state has no
# separate 'history states' to featurize individually"). That reasoning
# doesn't actually hold: _FourierMLPHistoryDeltaBody's per-history-state
# featurization trivially degenerates to a single state at n_history=1
# (nothing to concatenate across), which IS exactly the markovian case.
# Implemented by relaxing _FourierMLPHistoryDeltaBody's n_history>=2
# check to n_history>=1, adding a markovian dispatch branch in
# LatentPropagator.__init__ that builds this body with n_history=1
# (hardcoded, not cfg.n_history -- meaningless in markovian mode), and
# teaching step_one to reshape its (B, d_latent) input to (B, 1,
# d_latent) before calling the body (mirroring step_history's existing
# per-backbone body_input convention). Removed the now-incorrect
# mode='markovian'+backbone='fourier_mlp' validation raise in
# ks_latent/config.py. See ks_latent/models/propagator.py's
# _FourierMLPHistoryDeltaBody and LatentPropagator.step_one docstrings.
# Verified via 8 new unit tests in
# tests/unit/test_propagator_fourier_mlp.py (construction/shape,
# identity-at-init, rollout via the uniform step/rollout interface -- NOT
# step_history/rollout_history, which markovian mode never uses --
# masked+Fourier two-network split, fourier_ifft_readout, gradient flow,
# bfloat16 autocast), full existing unit suite re-run clean (no
# regressions), plus a real 4-epoch Stage 1 + 4-epoch Stage 2 warm-start
# smoke run of this exact recipe (both completed without error; smoke
# artifacts deleted before this real launch).
#
# "Put more parameters in the frequency-coefficients part, fewer in the
# real-function-value part": backbone='fourier_mlp' already structurally
# splits into two independent, SUMMED sub-networks when --prop-attn-window
# is finite -- masked_body (raw latent values, circular-band-masked,
# DIMENSION-PRESERVING at width=d_latent=44 regardless of --aux-hidden,
# per _MaskedMLPDeltaBody's docstring) and fourier_body (rfft features,
# width=--aux-hidden, explicitly irfft'd back to state space via
# --prop-fourier-ifft/FourierIFFTBody). Verified by direct instantiation
# this turn (hidden=480, n_blocks=2, attn_window=4): masked_body=11,880
# params vs. fourier_body=971,086 params -- an 81.7x skew toward the
# frequency path already, before even accounting for attn_window=4 being
# a genuinely restrictive raw-value window (d_latent=44 saturates to
# fully-dense at attn_window>=22, so 4 keeps the raw path meaningfully
# local, unlike Section 75/81's own attn_window=22 which was already
# fully dense). No change needed beyond picking these hyperparameters --
# the two-network architecture already gives exactly the requested
# asymmetry. AE (ViT, Section 52's exact structure) = 816,342 params,
# unchanged from 52. Propagator body total = 982,966 params (vs. 52's
# plain mlp/markovian propagator, size not directly comparable -- 52 used
# --aux-backbone mlp with its own default sizing, not verified against
# this run's fourier_mlp body).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section84_vit52struct_fouriermlpmarkovian_wspatial0135_200ep

echo "=== [1/3] Stage 1: ViT encoder/decoder (Section 52's exact structure: d_model=96, pos_encoding=linear, attn_window=4, token_window=16) + fourier_mlp/markovian propagator (masked raw-value path attn_window=4 + dense Fourier path hidden=480/n_blocks=2, fourier_ifft_readout), w_decorr=0, w_var=0.02, w_spatial=0.0135 (SIGNED, =0.01*1.35), w_var_floor=0, w_logdet=0.0035, NO delta_cap/lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone fourier_mlp --mode markovian \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --aux-hidden 480 --aux-blocks 2 --prop-attn-window 4 --prop-fourier-ifft \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.0135 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own fourier_mlp/markovian aux propagator, --amp, k_max=12, 300 epochs (Section 52's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 84 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
