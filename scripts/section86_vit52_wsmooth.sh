#!/bin/zsh
# User-directed 2026-09-04, "Section 86": "I still think 52 is the best.
# can we repeat 52, but add w_smooth = .001 and increase w_spatial 1.3x."
#
# Base: Section 52 (scripts/section52_spatial_signed_tuned5.sh) -- ViT
# (d_model=96 default, pos_encoding=linear, attn_window=4,
# token_window=16) encoder/decoder, mlp/markovian aux propagator,
# w_var=0.02, w_decorr=0, w_var_floor=0, w_logdet=0.0035, NO lambda_z/
# delta_cap, --amp, Stage-2 warm-started k_max=12 curriculum
# (k_warmup_epochs=210/k_mid=8/k_mid_epochs=175), 200ep Stage 1/300ep
# Stage 2. UNCHANGED from 52 except:
#   1. w_spatial (signed): 0.01 -> 0.013 (x1.3).
#   2. w_smooth: 0 -> 0.001 (NEW regularizer, added earlier this session --
#      ks_latent.training.losses.temporal_smoothness_loss, the
#      differentiable version of scripts/analyze_latent_smoothness.py's
#      real-trajectory step-size/curvature diagnostic). Section 85 (=
#      Section 81 + w_smooth=0.00075 + other changes) already showed
#      w_smooth improving both rollout MSE AND every smoothness measure
#      simultaneously relative to 81 with no collapse symptoms -- this is
#      the first test of w_smooth on Section 52's markovian-mode/mlp-
#      propagator architecture instead of the fourier_mlp/history-mode
#      lineage, at a slightly higher weight (0.001 vs 0.00075).
#      smooth_curvature_weight left at its default (1.0).
#
# Why 52 specifically: per the user, Section 52 (ViT + mlp/markovian
# propagator) has consistently had the best raw rollout MSE
# (best_val_kmax_mse=0.0256, the best in the whole project) and DA skill
# (skill_free_over_da=3.123, also the best), DESPITE catastrophic latent
# covariance conditioning (cond#=1.59e6) -- and
# scripts/analyze_latent_smoothness.py found 52 was ALSO the smoothest
# embedding of the whole comparison set by a clear margin (curvature
# 0.086 vs. 75/81's ~0.125-0.129), a second independent falsification of
# conditioning as a smoothness proxy. Section 86 tests whether pushing
# w_spatial slightly higher plus the new w_smooth term can push 52's
# already-best rollout/DA numbers and already-best smoothness even
# further, on its own terms rather than the fourier_mlp/hybrid lineage.
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.151979, no errors) and a 4-epoch Stage 2 warm-start smoke run
# (val_kmax_mse trending down 0.62->0.58 by epoch 3, expected trajectory
# shape this early) under this exact flag combination -- first real
# integration test of w_smooth with mode="markovian"/aux-backbone="mlp"
# at Section 52's real (non-smoke-profile) architecture size. No code
# changed this turn. Smoke artifacts cleaned up before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section86_mlpmarkovian_wvar002_wspatial0013_wsmooth0001_logdet0035_200ep

echo "=== [1/3] Stage 1: mlp/markovian aux (Section 52's exact architecture), w_decorr=0, w_var=0.02, w_spatial=0.013 (SIGNED, x1.3), w_var_floor=0, w_logdet=0.0035, w_smooth=0.001 (NEW), NO delta_cap/lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone mlp --mode markovian \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.013 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --w-smooth 0.001 \
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

echo "=== Section 86 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
