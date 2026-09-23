#!/bin/zsh
# User-directed 2026-09-03, "Section 67": follow-up to Section 66
# (fourier_mlp as encoder/decoder/propagator, Section 65's heavier
# regularizer weights -- rollout final relative RMSE=1.53, notably worse
# than Section 65's vit-AE+fourier_mlp-propagator combo's 0.98, despite
# better val_kmax_mse=0.0528 and vastly better conditioning cond#=286.7).
# The user asked: "yeah let's queue another section using exactly section
# 52's regularization terms with all fourier_mlp models" -- a controlled
# test isolating whether Section 66's worse rollout accuracy was caused
# by the heavier regularizer weights (shared with Section 65) rather than
# the fourier_mlp architecture itself, since Sections 65/66 held
# regularization constant and only varied architecture.
#
# Same model sizes/architecture as Section 66 (fourier_mlp encoder+
# decoder hidden=224/n_blocks=3 -> 811628 params; fourier_mlp/history2
# propagator hidden=140/n_blocks=2 -> 111344 params, same propagator
# reused in Phase 1 and 2 via --full-propagator + --init-prop-checkpoint
# warm-start), but regularizer weights switched to Section 52's exact
# recipe: w_var=0.02 (up from 0.015), w_spatial(signed)=0.01 (down from
# 0.035), w_logdet=0.0035 (down from 0.005), NO lambda_z (Section 52
# never used it -- omitted entirely here, not just set to 0, since
# --reg-start-epoch/RegConfig only matter when lambda_z/lambda_decorr are
# actually nonzero). w_decorr=0, w_var_floor=0, NO delta_cap -- unchanged
# family constants.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section67_fouriermlp_everything_section52regs_200ep

echo "=== [1/3] Stage 1: fourier_mlp encoder+decoder (hidden=224/n_blocks=3) + fourier_mlp/history2 aux propagator (hidden=140/n_blocks=2, full-propagator sizing), Section 52's regularizers (w_decorr=0, w_var=0.02, w_spatial=0.01 (SIGNED), w_var_floor=0, w_logdet=0.0035, NO lambda_z), NO delta_cap, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder fourier_mlp --aux-backbone fourier_mlp --mode history --n-history 2 \
  --fourier-mlp-hidden 224 --fourier-mlp-blocks 3 \
  --aux-hidden 140 --aux-blocks 2 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own (fourier_mlp/history2) aux propagator, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 67 (fourier_mlp everywhere, Section 52's regularizers) complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
