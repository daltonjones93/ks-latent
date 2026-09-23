#!/bin/zsh
# User-directed 2026-09-04, "Section 85": "can we revisit 81, except set
# lambda_z to 0, slightly increase w_spatial_signed by 1.2x and add the
# w_smooth term (maybe at .00075) and increase w_var 2x." (Clarified via
# question: "w_spatial term" in the original message meant w_smooth --
# w_spatial is already active in 81's recipe, adding a second one made no
# sense; w_smooth is the temporal-smoothness regularizer built earlier
# this session and flagged for testing "at a small weight with full
# Gate 3/4 monitoring" -- 0.00075 is exactly that.)
#
# Base: Section 81 (scripts/section81_vit_fourier_ifft_hybrid.sh) --
# vit_fourier_hybrid AE (ViT d_model=92, pos_encoding=linear,
# attn_window=4, token_window=16 + FourierIFFTBody fourier_hidden=270/
# fourier_blocks=2, fourier_kind=ifft, 1,509,438 params) + Section 75's
# exact fourier_mlp/history(n_history=2) propagator (attn_window=22,
# fourier_ifft_readout, hidden=480/n_blocks=2). UNCHANGED from 81 except:
#   1. lambda_z: 0.0002 -> 0 (dropped entirely, --lambda-z/--reg-start-epoch
#      flags removed from the command below). CORRECTION found this turn:
#      --lambda-z maps directly to RegConfig.lambda_z
#      (ks_latent/training/regularizer.py's BandedSmoothness) -- the SAME
#      banded-latent mechanism flagged in persistent memory as "never
#      propose/enable, known to collapse the latent" (reference project:
#      collapse observed at lambda_z >= 5e-3). Sections 75/81/82 all had a
#      small amount of it active (0.0002-0.0003, well below that
#      threshold) -- an earlier handoff doc's claim that this arc's
#      --lambda-z was "a different, log-det-related field, not the banded
#      one" was incorrect. Dropping it to 0 here removes that flagged
#      mechanism entirely, not just reduces it.
#   2. w_spatial (signed): 0.01 -> 0.012 (x1.2).
#   3. w_var: 0.01 -> 0.02 (x2).
#   4. w_smooth: 0 -> 0.00075 (NEW regularizer, added this session --
#      ks_latent.training.losses.temporal_smoothness_loss, the
#      differentiable version of scripts/analyze_latent_smoothness.py's
#      real-trajectory step-size/curvature diagnostic. Motivation:
#      Section 82's GIF-visible "huge jumps" were confirmed quantitatively
#      by that diagnostic, and w_spatial/D7/D8 regularize a DIFFERENT axis
#      entirely (cross-sectional channel correlation, not z_t vs.
#      z_{t+1}). Its own docstring flags a genuine collapse-to-a-point
#      risk in isolation -- 0.00075 is a small first test, paired here
#      with w_var raised 2x as extra anti-collapse counter-pressure.
#      smooth_curvature_weight left at its default (1.0)).
# Everything else (w_decorr=0, w_var_floor=0, w_logdet=0.008, --amp,
# --full-propagator, Stage-2 warm-started k_max=12 curriculum, 200ep
# Stage 1 / 300ep Stage 2) is identical to 81.
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.009089, no errors under this exact flag combination -- first real
# integration test of w_smooth with the production vit_fourier_hybrid/
# history-mode architecture, previously only smoke-tested against the
# toy vit+mlp/markovian combo) and a 4-epoch Stage 2 warm-start smoke run
# (val_kmax_mse reached 0.0645 by epoch 3, a healthy early trajectory).
# No code changed this turn beyond what was already added/tested for
# w_smooth earlier this session (8 passing unit tests, full suite clean)
# -- this run is a pure hyperparameter change. Smoke artifacts cleaned up
# before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section85_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_wvar002_wspatial0012_wsmooth00075_nolambdaz_200ep

echo "=== [1/3] Stage 1: Section 81's exact architecture (ViT+FourierIFFTBody hybrid AE, fourier_mlp/history2 propagator) with w_var=0.02 (x2), w_spatial=0.012 (SIGNED, x1.2), w_smooth=0.00075 (NEW), lambda_z REMOVED (was 0.0002), w_logdet=0.008 unchanged, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit_fourier_hybrid --aux-backbone fourier_mlp --mode history --n-history 2 \
  --d-model 92 --pos-encoding linear --attn-window 4 --token-window 16 --pool mean \
  --vit-fourier-fourier-hidden 270 --vit-fourier-fourier-blocks 2 --vit-fourier-kind ifft \
  --prop-attn-window 22 --prop-fourier-ifft \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.012 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.00075 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own fourier_mlp/history2 aux propagator (Section 75/81's exact architecture), --amp, k_max=12, 300 epochs (Section 75/81's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 85 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
