#!/bin/zsh
# User-directed 2026-09-02, "Section 62": "the same exact models and
# training settings as 52, but just add lambda_z = .0005 ... and only
# add it after the first 30 epochs" -- direct test of the banded
# latent-index-smoothness penalty (ks_latent/training/regularizer.py)
# against the "spikiness across latent index" seen in the
# latent_state_evolution GIFs. lambda_z alone (NO lambda_decorr) is a
# DELIBERATE choice here, not an oversight: the user was told explicitly
# that BandedSmoothness alone has a degenerate global optimum (every
# channel collapses to the constant vector) and lambda_decorr is the
# module's designed counterweight, and chose to run lambda_z alone anyway
# to directly observe that failure mode. --reg-start-epoch 30 (NEW CLI
# flag, wired to RegConfig.start_epoch, added today) means the AE trains
# completely unconstrained by this term for the first 30 epochs, then it
# activates at full weight (no ramp) from epoch 30 on.
# Otherwise identical to Section 52's recipe (default-size vit AE +
# mlp/markovian aux, w_decorr=0, w_var=0.02, w_spatial=0.01 (SIGNED),
# w_var_floor=0, w_logdet=0.0035, NO delta_cap, --amp, Stage 2
# warm-started, k_max=12, Stage 1 200ep/Stage 2 300ep).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section62_mlpmarkovian_wvar002_wspatialsigned01_logdet0035_lambdaz00005after30ep_200ep

echo "=== [1/3] Stage 1: mlp/markovian aux, w_decorr=0, w_var=0.02, w_spatial=0.01 (SIGNED), w_var_floor=0, w_logdet=0.0035, lambda_z=0.0005 (active from epoch 30), NO delta_cap, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone mlp --mode markovian \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --lambda-z 0.0005 --reg-start-epoch 30 \
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

echo "=== [3/3] Stage 2: warm-started from Stage 1's own mlp/markovian aux, NO delta_cap, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 62 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
