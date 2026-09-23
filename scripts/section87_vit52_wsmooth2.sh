#!/bin/zsh
# User-directed 2026-09-04, "Section 87": "lets do 87 with w_spatial .019,
# w_logdet = .004, w_smooth = .002. same everything else." -- direct
# continuation of Section 86 (= Section 52 + w_spatial/w_smooth), further
# pushing all three regularizer weights up from 86's values.
#
# Base: Section 86 (scripts/section86_vit52_wsmooth.sh), itself Section 52
# (ViT d_model=96 default, pos_encoding=linear, attn_window=4,
# token_window=16, mlp/markovian aux propagator, --amp, Stage-2
# warm-started k_max=12 curriculum, 200ep Stage 1/300ep Stage 2).
# UNCHANGED from 86 except:
#   1. w_spatial (signed): 0.013 -> 0.019.
#   2. w_logdet: 0.0035 -> 0.004.
#   3. w_smooth: 0.001 -> 0.002 (x2).
# w_var stays at 0.02 (52/86's value, unchanged), w_var_floor=0,
# w_decorr=0, no lambda_z (never used in the 52 lineage).
#
# Context: Section 86 (w_spatial=0.013, w_smooth=0.001) came in slightly
# WORSE than plain Section 52 on both best_val_kmax_mse (0.0283 vs 52's
# 0.0256) and conditioning (cond#=3.62e6 vs 52's 1.59e6), though
# scripts/analyze_latent_smoothness.py showed 86's ENCODED trajectories
# were measurably smoother than 52's (step/curvature both down) even as
# its trained PROPAGATOR's own Jacobian norm got slightly rougher. 87
# tests whether pushing all three weights further either recovers/beats
# 52's rollout number while keeping (or improving) 86's smoothness gain,
# or continues the same tradeoff pattern seen going from 75/81 to
# 76/77/82 (higher w_spatial hurting learnability past some point).
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.241383, no errors) and a 4-epoch Stage 2 warm-start smoke run
# (val_kmax_mse trending down 0.419->0.405 by epoch 3) under this exact
# flag combination. No code changed this turn. Smoke artifacts cleaned
# up before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section87_mlpmarkovian_wvar002_wspatial0019_wsmooth0002_logdet004_200ep

echo "=== [1/3] Stage 1: mlp/markovian aux (Section 52's exact architecture), w_decorr=0, w_var=0.02, w_spatial=0.019 (SIGNED), w_var_floor=0, w_logdet=0.004, w_smooth=0.002, NO delta_cap/lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone mlp --mode markovian \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.019 --spatial-signed --w-var-floor 0 --w-logdet 0.004 \
  --w-smooth 0.002 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own mlp/markovian aux, NO delta_cap, --amp, k_max=12, 300 epochs (Section 52's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 87 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
