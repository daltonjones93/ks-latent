#!/bin/zsh
# User-directed 2026-09-04, "Section 78": direct test of two ideas from
# the "why can't we get lower val_kmax_mse for 76/77" investigation --
# (1) "what if we increased the size of latent space? this might give us
# little more wiggle room for more smoothness but still having the same
# D_KY" and (2) "is w_var a better counterbalance for w_spatial or
# w_logdet?" (answer given: w_logdet, mechanistically -- it penalizes the
# FULL covariance's log-det, catching correlation-driven collapse, unlike
# w_var's marginal-per-channel-only mechanism -- see logdet_barrier_loss's
# and decorr_var_loss's docstrings).
#
# Diagnosis that motivated this (scripts/analyze_75_76_77_latent_geometry.py):
# w_spatial(signed) collapses the latent's EFFECTIVE dimensionality
# (participation ratio) by rewarding cross-channel correlation --
# Section 75 (w_spatial=0.01): cond#=21, participation ratio=14.2,
# encoder adds ZERO extra sensitivity beyond KS's own intrinsic chaos.
# Section 77 (w_spatial=0.03, same w_var/w_logdet as 75): cond#=122,
# participation ratio=3.6. Section 76 (w_spatial=0.04, w_var WEAKENED to
# 0.0005): cond#=462, participation ratio=2.4, encoder actively makes
# nearby physical states diverge 13% FASTER than the true physics itself
# once encoded -- a latent space locally rougher than the system it
# encodes, which no propagator architecture (we tried real MLP, masked
# MLP, complex-valued Fourier-IFFT) can smoothly fit.
#
# New capability: --d-latent (train_stage1_patched.py CLI flag) --
# d_latent was hardcoded to 44 everywhere in this session until now.
# w_spatial's spatial_coherence_loss operates over a FIXED ABSOLUTE
# bandwidth (spatial_bandwidth=3.0, index units), so a larger d_latent
# means the same absolute smoothing pressure eats a smaller fraction of
# the space, leaving more room for KS's true positive-Lyapunov directions
# to each get an effectively-independent latent direction.
#
# This run: d_latent 44->64 (~1.45x), encoder attn_window=8 held at its
# previous ABSOLUTE value (unchanged -- same local-window mechanism, now
# covering a smaller fraction of the larger ring), propagator/decoder
# window raised 22->32 (= new d_latent//2, the saturation point --
# verified directly: mask is all-True at window=32/d_latent=64, keeping
# them "effectively fully dense" exactly as in Sections 75-77, not
# reintroducing propagator/decoder masking as a new confound).
# Regularizers: w_var held at Section 75's 0.01 (isolating w_logdet as
# the primary counterbalance being tested, per the mechanistic argument
# above), w_spatial raised only modestly (0.01->0.015, 1.5x, NOT 77's 3x),
# w_logdet raised more than proportionally (0.008->0.016, 2x) as the
# counterbalance, lambda_z held at Section 75's proven-safe 0.0002.
#
# Param counts verified this turn by direct instantiation: AE (d_latent=64,
# hidden=224/n_blocks=4, attn_window=8/dec_attn_window=32) -> 1,056,452
# params. Propagator (d_latent=64, hidden=480/n_blocks=2, attn_window=32)
# -> 1,046,946 params -- both close to Sections 75-77's ~1M (some growth
# expected/accepted from the larger d_latent itself, not controlled for).
#
# Verified this turn before launch: a real 4-epoch Stage 1 smoke run
# (val_recon_final=0.008451 -- notably tighter than Sections 75-77's
# ~0.02 at 4 epochs, likely just the extra capacity from more latent
# channels) confirming d_latent=64/attn_window=8/dec_attn_window=32 on
# the saved AE config and attn_window=32 on the saved propagator config;
# a Stage 2 warm-start smoke run over 4 epochs with the extended
# k_max=20/500-epoch curriculum + --compare-k 12 all working correctly
# together. Full unit+integration suite re-run (new --d-latent CLI flag)
# -- see /tmp/full_test_run22.log.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section78_dlatent64_enc8_prop32_dec32_fourierifftreadout_wspatial015_wlogdet016_lambdaz00002_200ep

echo "=== [1/3] Stage 1: d_latent=64 (up from 44), masked fourier_mlp encoder (attn_window=8) + Fourier-IFFT-readout everywhere (decoder/propagator window=32, effectively dense at this d_latent), w_decorr=0, w_var=0.01 (held at Section 75's level), w_spatial=0.015 (SIGNED, 1.5x Section 75's 0.01), w_var_floor=0, w_logdet=0.016 (2x Section 75's 0.008, the primary counterbalance under test), lambda_z=0.0002 (Section 75's proven-safe value), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --d-latent 64 --encoder fourier_mlp --aux-backbone fourier_mlp --mode history --n-history 2 \
  --attn-window 8 --prop-attn-window 32 --dec-attn-window 32 --fourier-ifft-readout --prop-fourier-ifft \
  --fourier-mlp-hidden 224 --fourier-mlp-blocks 4 \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.01 --w-spatial 0.015 --spatial-signed --w-var-floor 0 --w-logdet 0.016 \
  --lambda-z 0.0002 --reg-start-epoch 0 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Latent covariance spectrum + D7/D8 bandedness check (also reports condition number and participation ratio for direct comparison against Sections 75/76/77's analyze_75_76_77_latent_geometry.py numbers) ==="
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own dense@window32+fourier_ifft_readout fourier_mlp/history2 aux propagator, --amp, k_max=20, 500 epochs, --compare-k 12 (directly comparable to Section 75's best_val_kmax_mse=0.0392) ==="
STAGE2_TAG="${TAG}_warmstart_k20_500ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 500 --k-max 20 --k-warmup-epochs 350 --k-mid 13 --k-mid-epochs 290 --compare-k 12 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 78 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
