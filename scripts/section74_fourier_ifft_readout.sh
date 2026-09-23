#!/bin/zsh
# User-directed 2026-09-03, "Section 74" (SUPERSEDES the first version of
# this script, which was killed mid-run due to a misunderstanding --
# "kill the run, since you misunderstood me. I want the propagator and
# the decoder to have the same structure as the encoder you described,
# but I want the attn_window to be much larger. that's what I meant by
# dense."). Original request: "apply an inverse fft to the frequency
# component output of the fourier mlp to map back to state space in the
# encoder and propagator and decoder ... don't mess with the masked mlp
# that doesn't interact with the fourier coefficients."
#
# What was wrong the first time: I implemented "dense for the propagator
# and decoder" as DROPPING the raw-value path entirely (pure
# Fourier+irfft, no masked_body/decoder_masked at all). The user's actual
# intent: propagator and decoder should have the SAME two-network
# structure as the encoder -- a masked raw-value path SUMMED with the
# Fourier-ifft path, non-interacting, exactly like the encoder -- just
# with a MUCH LARGER window than the encoder's 8, not no window at all.
#
# Fixed this turn: FourierMLPAutoencoderConfig gained `dec_attn_window`
# (independent of `attn_window`, which is encoder-only) so the decoder
# now always builds a masked raw-value path under fourier_ifft_readout
# (falling back to `attn_window`'s value if `dec_attn_window` isn't
# given). The propagator's `_FourierMLPHistoryDeltaBody` already built
# the correct two-network structure whenever `mask_window` was not
# None -- the bug there was purely at the CLI/launch level (--prop-dense
# forced None); now uses new --prop-attn-window (independent of
# --attn-window) instead. See FourierMLPAutoencoderConfig.
# fourier_ifft_readout's and PropagatorConfig.fourier_ifft_readout's
# docstrings (both corrected this turn) for the exact structure.
#
# Window sizes: encoder=8 (as originally requested), propagator/decoder=22
# (= d_latent // 2 for d_latent=44 -- the maximum MEANINGFUL circular-band
# radius on this ring: at this radius every latent index is already within
# window of every other, since the ring's farthest possible pairwise
# distance is exactly 22, so this is "as large as attn_window can get" via
# --prop-attn-window/--dec-attn-window while still going through the
# identical masked-path code as the encoder -- chosen as the clearest
# literal reading of "much larger", not an arbitrary guess).
#
# Architecture: encoder=fourier_mlp, masked_raw(attn_window=8) SUMMED with
# fourier_ifft path (using --fourier-ifft-readout). propagator=fourier_mlp,
# masked_raw(window=22) SUMMED with fourier_ifft path (--prop-fourier-ifft
# --prop-attn-window 22). decoder=fourier_mlp, masked_raw(window=22)
# SUMMED with fourier_ifft path (--dec-attn-window 22, via
# --fourier-ifft-readout). Same hidden/n_blocks sizing and Section 66's
# regularizers with lambda_z=0.002 as Sections 71-73.
#
# Param counts verified this turn by direct checkpoint inspection: AE
# (encoder masked@8 + decoder masked@22, both + fourier_ifft_readout) ->
# 1,002,332 params (essentially unchanged from Section 71's 1,001,432 --
# window size doesn't change allocated parameter count, only masking).
# Propagator (masked@22 + fourier_ifft_readout, hidden=480/n_blocks=2) ->
# 1,005,046 params.
#
# Verified this turn before launch: 4 corrected/new unit tests (2
# rewritten in test_autoencoder_fourier_mlp.py to check the two-network
# structure instead of the old wrong dense-only expectation; all 51 tests
# across the two fourier_mlp test files pass); full unit+integration
# suite (see /tmp/full_test_run18.log); a real 2-epoch Stage 1 smoke run
# (val_recon_final=0.090932 -- notably better than the first (wrong)
# version's 0.609826, since every component now has a raw-value path
# again) confirming attn_window=8/dec_attn_window=22 on the saved AE
# config and attn_window=22 on the saved propagator config, both with
# masked_body/decoder_masked/encoder_masked all present; a Stage 2
# warm-start smoke run over 4 epochs showed clean, monotonic loss decrease
# (val_kmax_mse 0.051->0.022). Smoke artifacts cleaned up before this real
# launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section74_enc8_prop22_dec22_fourierifftreadout_section66regs_lambdaz0002_200ep

echo "=== [1/3] Stage 1: fourier_mlp encoder (masked_raw@8 + fourier_ifft_readout) + fourier_mlp decoder (masked_raw@22 + fourier_ifft_readout) + fourier_mlp/history2 aux propagator (masked_raw@22 + fourier_ifft_readout, hidden=480/n_blocks=2, full-propagator sizing), w_decorr=0, w_var=0.015, w_spatial=0.035 (SIGNED), w_var_floor=0, w_logdet=0.005, lambda_z=0.002 (from epoch 0, 2x Section 66), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder fourier_mlp --aux-backbone fourier_mlp --mode history --n-history 2 \
  --attn-window 8 --prop-attn-window 22 --dec-attn-window 22 --fourier-ifft-readout --prop-fourier-ifft \
  --fourier-mlp-hidden 224 --fourier-mlp-blocks 4 \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.015 --w-spatial 0.035 --spatial-signed --w-var-floor 0 --w-logdet 0.005 \
  --lambda-z 0.002 --reg-start-epoch 0 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own masked@22+fourier_ifft_readout fourier_mlp/history2 aux propagator, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 74 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
