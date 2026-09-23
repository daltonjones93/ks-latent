#!/bin/zsh
# User-directed 2026-09-06, "Section 99": "okay, so I want to model this
# with a pde, but I think that this may require more smoothness. in
# order to get this smoothness, I want to increase d_latent to 96, and
# increase w_smooth 8x, w_spatial 2x from the 98 settings. Please run
# this as section 99, but keep everything else the same."
#
# Identical to Section 98 (scripts/section98_no_fourier_branch.sh, plain
# vit encoder/decoder, NO Fourier branch, d_model=56, pool=token_mlp/
# dec_pool=token_mlp reduction=8/hidden=128, attn_window=4/
# pos_encoding=linear/token_window=16, Section 52's EXACT mlp/markovian
# propagator hidden=128/n_blocks=3, w_var=0.02/w_var_floor=0/
# w_logdet=0.008/NO lambda_z/w_decorr=0) EXCEPT:
#   1. --d-latent 96 (up from the project's canonical default 44).
#   2. w_spatial (signed): 0.04 -> 0.08 (2x).
#   3. w_smooth: 0.003 -> 0.024 (8x).
#
# IMPORTANT CAVEAT (flagged before launch, per user request to run this
# regardless): the propagator's hidden=128/n_blocks=3 sizing is NOT
# scaled up to match d_latent=96 -- "keep everything else the same" was
# explicit, so this is exactly Section 52's original propagator capacity
# applied to a latent more than double the project's canonical 44. Every
# previous d_latent-growth experiment in this project (Sections 78, 82,
# 88) hit a real, never-fully-isolated propagator-capacity confound when
# the encoder grew but the propagator didn't -- if this run looks
# capacity-starved rather than genuinely smoother, that confound (not
# insufficient w_smooth/w_spatial) is the most likely explanation.
#
# Quick magnitude sanity check this turn (on Section 98's own checkpoint,
# d_latent=44 -- the real number will differ once trained at d_latent=96):
# l_smooth raw=0.0556, at w_smooth=0.024 -> contributes ~0.00133, roughly
# 8.6x l_recon's own raw magnitude (0.000155) -- a genuinely dominant
# term, consistent with the user's explicit intent to push smoothness
# hard for PDE-modeling purposes, not a miscalibration.
#
# Verified this turn: a real 4-epoch Stage 1 smoke run (val_recon_final=
# 0.033919, no errors -- confirms d_latent=96 + token_mlp pooling +
# these regularizer weights all construct and train without error) and a
# 4-epoch Stage 2 warm-start smoke run (val_kmax_mse trending down
# 0.74->0.61 by epoch 3, note the propagator's input/output projections
# automatically scale to d_latent=96 even though hidden/n_blocks stay
# Section 52's original 128/3). No code changed this turn (pure
# hyperparameter combination of existing capabilities). Smoke artifacts
# cleaned up before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section99_vitonly_dlatent96_dmodel56_tokenmlp_propmlpmarkovian52_wspatial08_wsmooth024_200ep

echo "=== [1/3] Stage 1: PLAIN vit encoder+decoder (NO Fourier branch, d_latent=96 UP FROM 44, d_model=56, pool=token_mlp/dec_pool=token_mlp reduction=8/hidden=128) + Section 52's EXACT mlp/markovian propagator (hidden=128/n_blocks=3, NOT scaled for the larger d_latent), w_decorr=0, w_var=0.02, w_spatial=0.08 (SIGNED, 2x Section 98's 0.04), w_var_floor=0, w_logdet=0.008, w_smooth=0.024 (8x Section 98's 0.003), NO lambda_z, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --d-latent 96 --encoder vit --aux-backbone mlp --mode markovian \
  --d-model 56 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.08 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.024 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own mlp/markovian aux, --amp, k_max=12, 300 epochs (Section 52's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 99 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
