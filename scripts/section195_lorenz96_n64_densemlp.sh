#!/bin/zsh
# User-directed 2026-09-22: "Let's also try training using the system of
# size 64, with latent dimension 16. the forcing constant should be 8."
# Companion to Section 194 (N=256, d_latent=96) -- same encoder family
# (local_field) and propagator (dense MLP, markovian), at a much smaller
# scale, to see how the approach performs away from the "match KS's own
# NX=256" scale.
#
# Encoder sizing: n_sites=8, local_channels=2 -> d_latent=16 (exactly the
# requested value), preserving the SAME "8 physical sites per patch"
# convention Section 194's own n_sites=32 uses at NX=256 (256/32=8,
# 64/8=8) -- a deliberate, consistent design choice across both scales,
# not an arbitrary independent pick.
#
# System: Lorenz-96, N=64, F=8.0 (dataset generated 2026-09-22, same
# ks_latent/solver/lorenz96.py). TRUE reference spectrum measured
# directly before any training (full N=64-direction spectrum):
#   lambda1=1.744, n_positive=21/64 (32.8%), D_KY=43.3 (67.7% of N)
# -- note how closely n_positive/N and D_KY/N match Section 194's own
# N=256 case (33.6%/67.8%) -- direct empirical confirmation of Lorenz-96's
# own extensivity (a roughly constant density of chaos per site,
# independent of N), the same property that motivated using it as a KS
# comparison system at all.
#
# Stage 1 ONLY, --nx 64 (new flag, see Section 194's own script for why
# it was needed).
#
# Verified via a real (--profile full, --epochs 3) dry run of this exact
# recipe before this launch: no NaN, trains normally (recon
# 0.766 -> 0.546 over 3 epochs).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section195_lorenz96_n64_densemlp

echo "=== [1/2] Stage 1 ONLY: Lorenz-96 N=64, local_field encoder (n_sites=8, channels=2, d_latent=16), dense MLP propagator (markovian), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset artifacts/datasets/lorenz96_trajectories_n64.h5 --nx 64 \
  --encoder local_field \
  --local-field-n-sites 8 --local-field-channels 2 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone mlp --mode markovian \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.12 --spatial-signed --w-logdet 0.01 --w-smooth 0.012 \
  --w-channel-mean 0.01 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/2] Diagnostics: reconstruction quality, standalone + multi-IC rollout, real Lyapunov spectrum vs. true L96 N=64 reference (lambda1=1.744, n_positive=21/64, D_KY=43.3) ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$AUX')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('d_latent:', d_latent)

DATASET = 'artifacts/datasets/lorenz96_trajectories_n64.h5'
with h5py.File(DATASET,'r') as f:
    traj20 = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, N = traj20.shape
with torch.no_grad():
    z_all = ae.encode(traj20.reshape(n*T, N)).reshape(n, T, -1)
z0b = z_all[:,0,:]

with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=150)
finite = torch.isfinite(traj_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('multi-IC (20) rollout first non-finite step:', first_bad)
maxz_per_t = traj_roll.abs().amax(dim=(0,2))
for t in [0,5,10,20,40,60,90,120,149]:
    if t < traj_roll.shape[1]:
        print(' t=%d  max|z| across all 20 ICs: %.4f' % (t, maxz_per_t[t].item()))
final_states = traj_roll[:,-1,:]
pairwise = torch.cdist(final_states, final_states)
print('final-state pairwise distance (min excl. diag, max, mean):',
      (pairwise + torch.eye(20)*1e9).min().item(), pairwise.max().item(), pairwise.mean().item())

with h5py.File(DATASET,'r') as f:
    traj = torch.tensor(f['trajectories'][3], dtype=torch.float32)
with torch.no_grad():
    z01 = ae.encode(traj[:2]).numpy()
res = lyapunov_spectrum_latent_propagator(
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=400, qr_every=5,
    dt_snap=0.1, warmup_steps=50, seed=0, max_abs_state=1e3,
)
print('lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('D_KY: could not bracket --', e)
print('(true L96 N=64 reference: lambda1=1.744, n_positive=21/64, D_KY=43.3)')
" > artifacts/logs/lyapunov_${TAG}.log 2>&1
cat artifacts/logs/lyapunov_${TAG}.log

echo "=== Section 195 complete ==="
