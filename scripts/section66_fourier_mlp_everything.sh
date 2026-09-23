#!/bin/zsh
# User-directed 2026-09-03: "I would like to try section 66 using the
# fourier_mlp as the encoder, decoder and propagator. I would like to use
# roughly the same model sizes as section 52 as well as the same
# regularization parameters as 65. we will use the same propagator in
# phase 1 and 2."
#
# NOTE ON NUMBERING: "Section 66" collides with an already-written
# analysis-only doc entry (docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md's
# "66. Verified: the latent index shows genuine periodic (ring)
# structure..."), which is not a training run. Flagged to the user;
# proceeding with this tag as "section66" per their explicit request --
# if a doc write-up is done for this run, it should use a number after
# whatever the eventual 54-65 backfill lands on, not literally "66" in
# the file, to avoid two different things both being called Section 66.
#
# New capability implemented for this run: 'fourier_mlp' as an
# ENCODER/DECODER (ks_latent/models/autoencoder_fourier_mlp.py,
# KSAutoencoderFourierMLP; FourierMLPAutoencoderConfig in config.py) --
# raw NX physical values concatenated with a fixed real/imag rfft
# featurization, fed through a plain residual MLP (MLPDeltaBody, made
# public/reused from ks_latent.models.propagator -- the same class
# 'fourier_mlp' the PROPAGATOR backbone already used, and the same class
# every 'mlp' propagator variant uses). Mirror decoder. This generalizes
# the propagator-side 'fourier_mlp' idea (which won the Section 65
# propagator comparison: val_kmax_mse=0.0559 vs. 0.0684 for mlp/history5
# and a dynamically-collapsed 0.2574/D_KY=5.69 for 'fno_mlp') to the
# encoder/decoder role.
#
# Sizing ("roughly the same model sizes as section 52"): checked directly
# by instantiating both components and sweeping hidden/n_blocks --
#   encoder+decoder: hidden=224, n_blocks=3 -> 811628 params (Section
#     52's default ViT AE: 816342, ratio 0.994).
#   propagator (backbone=fourier_mlp, mode=history, n_history=2):
#     hidden=140, n_blocks=2 -> 111344 params (Section 52's default
#     Stage-2 mlp/markovian propagator: 111532, ratio 0.998).
#
# Regularizers ("the same regularization parameters as 65"): w_var=0.015,
# w_spatial(signed)=0.035, w_logdet=0.005, lambda_z=0.001 active from
# EPOCH 0 (reg_start_epoch=0, matching Section 65 exactly -- NOT delayed
# to epoch 30 like Sections 62-64). lambda_decorr stays 0 (lambda_z
# alone, same deliberate choice as throughout this sub-investigation).
# w_decorr=0, w_var_floor=0, NO delta_cap -- unchanged family constants.
#
# "the same propagator in phase 1 and 2": --full-propagator sizes Stage
# 1's aux propagator identically to the real Stage-2 propagator
# (hidden=140/n_blocks=2/backbone=fourier_mlp/mode=history/n_history=2),
# and Stage 2 warm-starts from it via --init-prop-checkpoint (which loads
# the ENTIRE saved PropagatorConfig, so no --backbone/--mode/etc need
# repeating there).
#
# Verified end-to-end before this launch (not just unit tests): a real
# 2-epoch Stage 1 smoke run with this exact --encoder/--aux-backbone/
# --mode/--n-history/sizing combination, confirmed checkpoint round-trip
# (KSAutoencoderFourierMLP, 811628 params; LatentPropagator backbone=
# fourier_mlp mode=history, 111344 params) and a successful Stage 2
# --init-prop-checkpoint warm-start.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section66_fouriermlp_everything_wvar0015_wspatialsigned0035_logdet0005_lambdaz0001fromepoch0_200ep

echo "=== [1/3] Stage 1: fourier_mlp encoder+decoder (hidden=224/n_blocks=3) + fourier_mlp/history2 aux propagator (hidden=140/n_blocks=2, full-propagator sizing), w_decorr=0, w_var=0.015, w_spatial=0.035 (SIGNED), w_var_floor=0, w_logdet=0.005, lambda_z=0.001 (from epoch 0), NO delta_cap, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder fourier_mlp --aux-backbone fourier_mlp --mode history --n-history 2 \
  --fourier-mlp-hidden 224 --fourier-mlp-blocks 3 \
  --aux-hidden 140 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.015 --w-spatial 0.035 --spatial-signed --w-var-floor 0 --w-logdet 0.005 \
  --lambda-z 0.001 --reg-start-epoch 0 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own (fourier_mlp/history2) aux propagator, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 66 (fourier_mlp everywhere) complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
