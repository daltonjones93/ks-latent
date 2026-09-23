#!/bin/zsh
# User-directed 2026-09-22: "can we try training on lorenz 96 using the
# 192 encoder and decoder, and a dense mlp propagator (markovian)."
#
# Encoder/decoder: EXACTLY Section 192's own local_field config
# (n_sites=32, channels=3, mix_radius=2, n_mix_layers=3, hidden=32 ->
# d_latent=96) -- freshly trained on Lorenz-96 data (KS-trained weights
# would be meaningless here; "the 192 encoder" means the ARCHITECTURE,
# not the checkpoint). Propagator: --aux-backbone mlp --mode markovian
# (a plain, fully-dense residual MLP -- no attention/local-window
# restriction, UNLIKE 192's own masked_mlp_expand) -- this project's own
# history (Section 168, KS) found a plain dense MLP propagator recovered
# genuine chaos where attention-windowed backbones collapsed; this tests
# whether that finding transfers to a structurally different chaotic
# system.
#
# System: Lorenz-96, N=256 (dataset generated 2026-09-22,
# ks_latent/solver/lorenz96.py -- see that module's docstring for the
# full solver/validation writeup, tests/unit/test_lorenz96.py for the
# 10-test validation suite). F=8.0 (Lorenz's own standard forcing).
# TRUE reference spectrum measured directly with this project's own
# Benettin/QR code before any training (full N=256-direction spectrum,
# guaranteed to bracket since trace(J)=-N exactly for any state):
#   lambda1=1.755, n_positive=86/256 (33.6%), D_KY=173.5 (67.8% of N)
#
# Stage 1 ONLY (matching this arc's own established "check before going
# further" discipline) -- --nx 64 for a different N would need
# regenerating the whole ae_cfg; --nx is a NEW flag added this session
# (train_stage1_patched.py previously hardcoded NX=256, never needed
# overriding since every section before this one trained on KS data).
#
# Verified via a real (--profile full, --epochs 3) dry run of this exact
# recipe before this launch: no NaN, trains normally (recon
# 0.709 -> 0.353 over 3 epochs).
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section194_lorenz96_n256_densemlp

echo "=== [1/2] Stage 1 ONLY: Lorenz-96 N=256, local_field encoder (Section 192's exact config, d_latent=96), dense MLP propagator (markovian), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset artifacts/datasets/lorenz96_trajectories_n256.h5 --nx 256 \
  --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
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

echo "=== [2/2] Diagnostics: reconstruction quality, standalone + multi-IC rollout, real Lyapunov spectrum vs. true L96 N=256 reference (lambda1=1.755, n_positive=86/256, D_KY=173.5) ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$AUX')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('d_latent:', d_latent)

DATASET = 'artifacts/datasets/lorenz96_trajectories_n256.h5'
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
print('(true L96 N=256 reference: lambda1=1.755, n_positive=86/256, D_KY=173.5)')
" > artifacts/logs/lyapunov_${TAG}.log 2>&1
cat artifacts/logs/lyapunov_${TAG}.log

echo "=== Section 194 complete ==="
