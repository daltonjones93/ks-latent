#!/bin/zsh
# User-directed 2026-09-22, following up on Section 195 (N=64, d_latent=16,
# F=8.0): that run showed a real, likely information-theoretic shortfall --
# d_latent=16 < true D_KY=43.3, so a 16-dim latent space cannot faithfully
# represent the true ~43-dimensional attractor. User: "is there a way
# through the forcing constant to reduce D_KY?" -- yes: F directly
# controls how chaotic Lorenz-96 is. Measured directly (this project's own
# Benettin/QR code, not a literature guess) via an F-sweep at N=64:
#   F=2,3: not chaotic (D_KY=0)      F=6: D_KY=35.8
#   F=4:   D_KY=5.3                  F=7: D_KY=40.3
#   F=4.2: D_KY=9.8 (quick estimate) F=8: D_KY=43.2 (Section 195's own F)
#   F=4.4: D_KY=0 (!) -- a narrow periodic window embedded in the
#          transition-to-chaos region, found directly by this sweep, NOT
#          monotonic in F.
#   F=4.6: D_KY=17.5   F=4.9: D_KY=2.3 (another narrow window)
# User: "can we try F=4.2?" -- comfortably BELOW d_latent=16 this time (an
# adequately-sized test, unlike 195's undersized one). A longer, more
# carefully converged re-measurement (80,000 steps, qr_every=20,
# warmup=10,000 -- F=4.2's dynamics are much slower, lambda1~0.08,
# Lyapunov time~12.5 model-time-units, needing much more integration time
# to converge than F=8's own lambda1~1.7 case) gives the reference used
# here:
#   lambda1=0.080, n_positive=5/64 (7.8%), D_KY=11.3 (17.7% of N)
#
# Dataset (generated 2026-09-22): spinup_time=200, trajectory_time=200
# (both increased from 195's own 50/100 -- F=4.2's much slower dynamics
# need proportionally more integration time to reach/sample the
# attractor; 200 model-time-units is ~16 Lyapunov times per trajectory,
# comparable in SPIRIT to how many Lyapunov times 195's own F=8 dataset
# covered at its own much faster timescale).
#
# Same encoder/propagator architecture as Section 195 (local_field
# n_sites=8/channels=2 -> d_latent=16, dense MLP propagator, markovian) --
# ONLY the dataset (F=4.2 instead of F=8) changes, isolating the forcing
# constant as the sole variable in this comparison.
#
# Standalone/multi-IC rollout extended to k=2000 (vs. 195's own k=150) --
# at F=4.2's own much slower timescale, 150 steps*dt_snap=0.1=15 model-
# time-units is only ~1.2 Lyapunov times (barely any dynamics), whereas
# 195's own 150 steps at F=8 covered ~26 Lyapunov times; k=2000 steps
# (200 model-time-units) covers ~16 Lyapunov times here, a much fairer
# comparison of "does this saturate/behave sensibly over many
# characteristic times."
#
# Verified via a real (--profile full, --epochs 3) dry run of this exact
# recipe before this launch: no NaN, and reconstruction converges MUCH
# faster than 195's own F=8 case (recon 0.037 after just 3 epochs vs.
# 195's 0.55) -- consistent with F=4.2's weaker/lower-dimensional dynamics
# being an easier target.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section196_lorenz96_n64_f4p2_densemlp

echo "=== [1/2] Stage 1 ONLY: Lorenz-96 N=64 F=4.2, local_field encoder (n_sites=8, channels=2, d_latent=16), dense MLP propagator (markovian), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset artifacts/datasets/lorenz96_trajectories_n64_f4.2.h5 --nx 64 \
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

echo "=== [2/2] Diagnostics: reconstruction quality, standalone + multi-IC rollout (k=2000, ~16 Lyapunov times), real Lyapunov spectrum vs. true L96 N=64/F=4.2 reference (lambda1=0.080, n_positive=5/64, D_KY=11.3) ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$AUX')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('d_latent:', d_latent)

DATASET = 'artifacts/datasets/lorenz96_trajectories_n64_f4.2.h5'
with h5py.File(DATASET,'r') as f:
    traj20 = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, N = traj20.shape
with torch.no_grad():
    z_all = ae.encode(traj20.reshape(n*T, N)).reshape(n, T, -1)
z0b = z_all[:,0,:]

with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=2000)
finite = torch.isfinite(traj_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('multi-IC (20) rollout first non-finite step:', first_bad)
maxz_per_t = traj_roll.abs().amax(dim=(0,2))
for t in [0,50,100,300,600,1000,1500,1999]:
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
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=2000, qr_every=10,
    dt_snap=0.1, warmup_steps=200, seed=0, max_abs_state=1e3,
)
print('lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('D_KY: could not bracket --', e)
print('(true L96 N=64/F=4.2 reference: lambda1=0.080, n_positive=5/64, D_KY=11.3)')
" > artifacts/logs/lyapunov_${TAG}.log 2>&1
cat artifacts/logs/lyapunov_${TAG}.log

echo "=== Section 196 complete ==="
