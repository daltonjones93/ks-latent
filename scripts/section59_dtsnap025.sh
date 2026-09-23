#!/bin/zsh
# User-directed 2026-09-02, "Section 59": "the exact same training as
# section 52, but use delta_t = .25 instead" -- Section 52's recipe
# (default-size vit AE + mlp/markovian aux, w_decorr=0, w_var=0.02,
# w_spatial=0.01 (SIGNED), w_var_floor=0, w_logdet=0.0035, NO delta_cap,
# --amp, Stage 2 warm-started, k_max=12, Stage 1 200ep/Stage 2 300ep),
# unchanged, EXCEPT --dt-snap 0.25 (physical time between snapshots,
# every dataset used elsewhere in this document is dt_snap=1.0 --
# see PropagatorConfig.dt_snap's docstring / train_stage1_patched.py's
# --dt-snap help). --dataset is pointed at a NEW path
# (stage1_trajectories_dtsnap025.h5) rather than the default
# stage1_trajectories_dtsnap1.h5 -- both train_stage1_patched.py and
# train_stage2_patched.py only auto-generate a dataset when the given
# --dataset path does not already exist, so reusing the default path
# would have silently loaded the existing dt_snap=1.0 data and made
# --dt-snap a no-op. snapshot_every = dt_snap/0.05 = 5 (vs. 20 at
# dt_snap=1.0), so this dataset has 4x as many snapshots per trajectory
# at the same physical trajectory_time=250.0 -- expect Stage 1 dataset
# generation to take noticeably longer than prior sections.
#
# NOTE: k_max=12 now covers 12*0.25=3.0 physical time units of rollout,
# vs. 12*1.0=12.0 in every other section's recipe -- a real difference in
# what's being asked of the propagator, not just a per-step step-size
# change, worth flagging when comparing val_kmax_mse/Gate 3/4 results
# against Section 52 (or any other prior section).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap025.h5
TAG=section59_mlpmarkovian_dtsnap025_wvar002_wspatialsigned01_logdet0035_200ep

echo "=== [1/3] Stage 1: dt_snap=0.25 (dataset: $DATASET), mlp/markovian aux, w_decorr=0, w_var=0.02, w_spatial=0.01 (SIGNED), w_var_floor=0, w_logdet=0.0035, NO delta_cap, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder vit --aux-backbone mlp --mode markovian \
  --pos-encoding linear --attn-window 4 --token-window 16 \
  --dt-snap 0.25 --dataset "$DATASET" \
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

with h5py.File('$DATASET','r') as f:
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

echo "=== [3/3] Stage 2: dt_snap=0.25, warm-started from Stage 1's own mlp/markovian aux, NO delta_cap, --amp, k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --dataset "$DATASET" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

echo "=== Section 59 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
