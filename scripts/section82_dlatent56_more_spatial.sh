#!/bin/zsh
# User-directed 2026-09-04, "Section 82": "I think the next test then,
# assuming 81 looks better than 75, is to increase w_spatial for 81 to
# .015 and lambda_z to .0003. we can slowly get smoother embeddings. we
# could try this with embedding dimension 56."
#
# Section 81 vs 75 at the same curriculum point (k_now=7): 81 showed
# val_kmax_mse~=0.035 vs. 75's 0.053 -- 81 confirmed better, condition
# satisfied.
#
# Same architecture as Section 81 (ViT d_model=92, Section 52 structure,
# SUMMED with FourierIFFTBody hidden=270/n_blocks=2, "only using the
# frequency components", fourier_kind="ifft"; propagator = same
# fourier_mlp/history2 family, hidden=480/n_blocks=2, fourier_ifft_readout),
# with three changes:
#   - d_latent: 44 -> 56. attn_window/dec_attn_window on the AE side are
#     UNAFFECTED (the ViT branch's --attn-window=4 is a TOKEN-space local
#     window, unrelated to d_latent; the "ifft" Fourier branch has no
#     masked/windowed path at all). The PROPAGATOR's --prop-attn-window
#     DOES need to track d_latent (it operates directly in d_latent-ring
#     space) -- raised 22->28 (=56//2, the new saturation point) to stay
#     "effectively fully dense," matching Section 78's precedent when
#     d_latent grew 44->64 (verified directly this turn: mask is all-True
#     at window=28/d_latent=56).
#   - w_spatial (signed): 0.01 -> 0.015 (1.5x).
#   - lambda_z: 0.0002 -> 0.0003 (1.5x).
# w_var/w_logdet/w_decorr/w_var_floor unchanged from Section 75/81's
# recipe (0.01/0.008/0/0).
#
# Param counts verified this turn by direct instantiation: AE (d_latent=56,
# d_model=92, fourier_hidden=270/n_blocks=2) -> 1,552,374 params (vit
# 792,278 + fourier_encoder 379,948 + fourier_decoder 380,148 -- still
# commensurate, ratio ~1.04). Propagator (d_latent=56, hidden=480/
# n_blocks=2, attn_window=28) -> 1,029,610 params.
#
# Verified this turn before launch: a real 4-epoch Stage 1 smoke run
# (val_recon_final=0.006143) and a 4-epoch Stage 2 warm-start smoke run
# -- val_kmax_mse=0.024543, an even stronger early signal than Section
# 81's own 4-epoch smoke result (0.0336). No code changed this turn (pure
# CLI/hyperparameter combination of existing capabilities), so the
# existing test suite is unaffected -- not re-run. Smoke artifacts
# cleaned up before this real launch.
#
# Queued to launch automatically once Section 81's Stage 2 (outer PID
# 92540/92542) finishes.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section82_dlatent56_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_wspatial015_lambdaz0003_200ep

echo "=== [1/3] Stage 1: vit_fourier_hybrid encoder/decoder (d_latent=56, ViT d_model=92 + FourierIFFTBody hidden=270/n_blocks=2, ifft kind) + fourier_mlp propagator (d_latent=56, attn_window=28, fourier_ifft_readout, hidden=480/n_blocks=2), w_decorr=0, w_var=0.01, w_spatial=0.015 (SIGNED, 1.5x Section 75/81's 0.01), w_var_floor=0, w_logdet=0.008, lambda_z=0.0003 (1.5x Section 75/81's 0.0002, from epoch 0), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --d-latent 56 --encoder vit_fourier_hybrid --aux-backbone fourier_mlp --mode history --n-history 2 \
  --d-model 92 --pos-encoding linear --attn-window 4 --token-window 16 --pool mean \
  --vit-fourier-fourier-hidden 270 --vit-fourier-fourier-blocks 2 --vit-fourier-kind ifft \
  --prop-attn-window 28 --prop-fourier-ifft \
  --aux-hidden 480 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.01 --w-spatial 0.015 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --lambda-z 0.0003 --reg-start-epoch 0 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own fourier_mlp/history2 aux propagator, --amp, k_max=12, 300 epochs (Section 75/81's curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 82 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
