#!/bin/zsh
# User-directed 2026-09-04, "Section 77": "I think lambda_z was too high
# in 76. can we run 77 with the same regularization parameters as 75,
# except increase w_spatial_signed 3x. run stage 2 for 500 epochs up to
# k = 20. same encoder, decoder and propagator. make attn_window larger
# by 1.5 times for the propagator and decoder."
#
# Investigated the attn_window request directly before implementing:
# d_latent=44, so the maximum possible circular ring distance is exactly
# 22 -- Section 75/76's propagator/decoder window=22 was ALREADY fully
# dense (verified directly: the mask is literally all-True at window=22,
# and stays all-True at 33/44/1000 -- there's no further effect past that
# saturation point). So "1.5x" (22->33) would have been a pure no-op.
# Flagged to the user, who chose: leave propagator/decoder dense (drop
# the window-scaling instruction entirely), only vary the regularizers.
#
# So: SAME architecture as Sections 75/76 (masked fourier_mlp
# encoder@attn_window=8 + Fourier-IFFT-readout, propagator/decoder
# window=22 -- i.e. effectively fully dense, unchanged), same size
# (~1M params each). Regularizers = Section 75's exact recipe (w_var=0.01,
# w_logdet=0.008, lambda_z=0.0002 -- reverted from Section 76's 0.0005,
# which the user suspects was too high) with ONLY w_spatial(signed)
# raised 3x: 0.01 -> 0.03.
#
# Stage 2 extended per user direction, same as Section 76: 500 epochs,
# k_max=20, same proportionally-scaled curriculum (k_warmup_epochs=350,
# k_mid=13, k_mid_epochs=290). Also carries forward --compare-k 12 (added
# 2026-09-04 for exactly this situation): pools MSE over just the first
# 12 steps of the same k_max=20 rollout, directly comparable to Section
# 75's own k_max=12 best_val_kmax_mse=0.0392.
#
# Verified this turn: a real 4-epoch Stage 1 smoke run
# (val_recon_final=0.022296) and a 4-epoch Stage 2 warm-start smoke run
# confirming the extended curriculum + --compare-k flags all work
# together correctly. No code changed this turn (pure hyperparameter
# change reusing the compare_k capability added earlier today), so the
# existing test suite is unaffected -- not re-run.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section77_enc8_prop22_dec22_fourierifftreadout_wspatialsigned03_3x_lambdaz00002_200ep

echo "=== [1/3] Stage 1: masked fourier_mlp encoder (attn_window=8) + Fourier-IFFT-readout everywhere (decoder/propagator window=22, effectively dense, same size as Section 75), w_decorr=0, w_var=0.01, w_spatial=0.03 (SIGNED, 3x Section 75's 0.01), w_var_floor=0, w_logdet=0.008, lambda_z=0.0002 (Section 75's value, down from Section 76's 0.0005), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder fourier_mlp --aux-backbone fourier_mlp --mode history --n-history 2 \
  --attn-window 8 --prop-attn-window 22 --dec-attn-window 22 --fourier-ifft-readout --prop-fourier-ifft \
  --fourier-mlp-hidden 224 --fourier-mlp-blocks 4 \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.01 --w-spatial 0.03 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own dense@window22+fourier_ifft_readout fourier_mlp/history2 aux propagator, --amp, k_max=20, 500 epochs, --compare-k 12 ==="
STAGE2_TAG="${TAG}_warmstart_k20_500ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 500 --k-max 20 --k-warmup-epochs 350 --k-mid 13 --k-mid-epochs 290 --compare-k 12 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 77 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
