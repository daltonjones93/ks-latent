#!/bin/zsh
# User-directed 2026-09-03, "Section 75": direct test of the lambda_z
# hypothesis from the "seems like we can't train an effective propagator
# with this setup. any ideas why?" discussion -- Sections 71/72/74 (all
# lambda_z=0.002, masked/Fourier-IFFT fourier_mlp architectures) plateaued
# at val_kmax_mse ~0.15-0.29 despite healthy Stage-1 diagnostics (tight
# reconstruction, no eigenvalue collapse, strong D7/D8 bandedness),
# while Section 66 (same family, lambda_z=0.001, UNMASKED) reached 0.053
# -- the one constant across every failing run was lambda_z doubled from
# 0.001, and RegConfig.lambda_z's own docstring documents measured
# collapse risk at lambda_z>=5e-3 when used (as in every one of these
# runs, 66 included) without its paired anti-collapse lambda_decorr term.
#
# This run: identical architecture to Section 74 (masked fourier_mlp
# encoder@attn_window=8 + Fourier-IFFT, masked fourier_mlp propagator/
# decoder@window=22 + Fourier-IFFT, ~1M params each), but with
# lambda_z cut 10x further than Section 66's own value (0.001 -> 0.0002)
# and three other regularizer weights adjusted per user direction:
# w_var 0.015->0.01, w_spatial(signed) 0.035->0.01, w_logdet 0.005->0.008.
#
# Verified this turn: a real 4-epoch Stage 1 + Stage 2 warm-start smoke
# run (val_recon_final=0.021599 after just 4 epochs; Stage 2
# val_kmax_mse 0.034->0.011 over 4 epochs -- much healthier early
# trajectory than Sections 71/72/74's equivalent smoke checks). No code
# changed this turn (pure hyperparameter change), so the existing test
# suite is unaffected -- not re-run.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section75_enc8_prop22_dec22_fourierifftreadout_wvar01_wspatialsigned01_logdet008_lambdaz00002_200ep

echo "=== [1/3] Stage 1: masked fourier_mlp encoder (attn_window=8) + Fourier-IFFT-readout everywhere (decoder/propagator masked@window=22 + Fourier-IFFT, --prop-dense-equivalent via --prop-attn-window 22 --prop-fourier-ifft), w_decorr=0, w_var=0.01, w_spatial=0.01 (SIGNED), w_var_floor=0, w_logdet=0.008, lambda_z=0.0002 (from epoch 0, 10x below Section 66's 0.001), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder fourier_mlp --aux-backbone fourier_mlp --mode history --n-history 2 \
  --attn-window 8 --prop-attn-window 22 --dec-attn-window 22 --fourier-ifft-readout --prop-fourier-ifft \
  --fourier-mlp-hidden 224 --fourier-mlp-blocks 4 \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.01 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own masked@22+fourier_ifft_readout fourier_mlp/history2 aux propagator, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 75 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
