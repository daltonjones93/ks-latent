#!/bin/zsh
# User-directed 2026-09-04, "Section 79": "design 79 to be exactly the
# same as 77 but increase w_pred 2x. so we can control variables."
#
# Tests the w_pred hypothesis from the "would increasing w_pred force the
# embedding to be more propagatable" discussion: Stage1TrainingConfig.
# w_pred (default 0.5, active in EVERY section this session including 77
# -- never explicitly set, so always at its default) weights l_pred, a
# loss comparing the co-trained --full-propagator aux propagator's
# k_pred=2-step rollout (decoded back to physical space) against the true
# future physical state. Since ae.parameters() (encoder AND decoder) and
# aux.parameters() are optimized JOINTLY on this loss, w_pred is a direct
# pressure on the ENCODER to arrange z so that ITS OWN co-trained
# propagator can roll it forward accurately -- literally "force the
# embedding to be more propagatable," already active by default the
# whole session, now doubled to test its strength as a counterbalance to
# w_spatial's collapse-inducing pull (see scripts/
# analyze_75_76_77_latent_geometry.py's diagnosis: Section 77's
# w_spatial=0.03 alone, with w_var/w_logdet held at Section 75's levels,
# already produced participation ratio 3.6 vs Section 75's 14.2).
#
# Single-variable change from Section 77: --w-pred 1.0 (2x the default
# 0.5), added -- every other flag identical to Section 77's script
# (architecture: d_latent=44, encoder attn_window=8, propagator/decoder
# window=22 effectively dense, fourier_ifft_readout everywhere, same
# ~1M-param sizing; regularizers: w_var=0.01, w_spatial=0.03 signed,
# w_logdet=0.008, lambda_z=0.0002; same extended Stage 2 curriculum,
# 500 epochs/k_max=20/--compare-k 12 for direct comparison against both
# Section 75's best_val_kmax_mse=0.0392 and Section 77's own numbers at
# the same k_now).
#
# Verified this turn: a real 4-epoch Stage 1 smoke run
# (val_recon_final=0.023321) confirming the checkpoint's architecture
# matches Section 77 exactly (d_latent=44, attn_window=22,
# fourier_ifft_readout=True), and a Stage 2 warm-start smoke run
# confirming the extended curriculum + --compare-k flags all work
# together correctly. No code changed this turn (--w-pred already
# existed as a CLI flag, never previously used in this session's
# fourier_mlp-family launches), so the existing test suite is unaffected
# -- not re-run.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section79_enc8_prop22_dec22_fourierifftreadout_wspatialsigned03_wpred10_2x_lambdaz00002_200ep

echo "=== [1/3] Stage 1: identical to Section 77, EXCEPT --w-pred 1.0 (2x the default 0.5) -- masked fourier_mlp encoder (attn_window=8) + Fourier-IFFT-readout everywhere (decoder/propagator window=22, effectively dense), w_decorr=0, w_var=0.01, w_spatial=0.03 (SIGNED, 3x Section 75's 0.01), w_var_floor=0, w_logdet=0.008, lambda_z=0.0002, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder fourier_mlp --aux-backbone fourier_mlp --mode history --n-history 2 \
  --attn-window 8 --prop-attn-window 22 --dec-attn-window 22 --fourier-ifft-readout --prop-fourier-ifft \
  --fourier-mlp-hidden 224 --fourier-mlp-blocks 4 \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.01 --w-spatial 0.03 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-pred 1.0 \
  --lambda-z 0.0002 --reg-start-epoch 0 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Latent covariance spectrum + D7/D8 bandedness check (also reports condition number and participation ratio for direct comparison against Section 77's numbers -- isolates whether w_pred alone recovers effective dimensionality) ==="
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own dense@window22+fourier_ifft_readout fourier_mlp/history2 aux propagator, --amp, k_max=20, 500 epochs, --compare-k 12 (directly comparable to Section 75's best_val_kmax_mse=0.0392 and Section 77's own numbers) ==="
STAGE2_TAG="${TAG}_warmstart_k20_500ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 500 --k-max 20 --k-warmup-epochs 350 --k-mid 13 --k-mid-epochs 290 --compare-k 12 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 79 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
