#!/bin/zsh
# User-directed 2026-09-22, following up on Section 196 (N=64, F=4.2,
# d_latent=16, genuinely chaotic but lambda1/D_KY both undershooting the
# true reference). User asked about the timestep: dt_snap=0.1 was reused
# unchanged from the F=8 dataset, but relative to each system's OWN
# Lyapunov time this is a very different resolution:
#   F=8.0:  dt_snap/tau ~ 0.1/0.58  ~ 17.5% of a Lyapunov time
#   F=4.2:  dt_snap/tau ~ 0.1/12.5  ~  0.8% of a Lyapunov time (~22x finer)
# Consecutive snapshots at F=4.2 were nearly identical -- exactly the
# "dt_snap too small" collapse-risk mechanism this project's own earliest
# history (KS representation-collapse postmortem) already flagged. User:
# "sure, do dt_snap = 1.0 and try the training again."
#
# NEW dataset (regenerated 2026-09-22): snapshot_every=100 (dt=0.01
# unchanged) -> dt_snap=1.0, an exact 10x increase from Section 196's own
# 0.1 (~8% of a Lyapunov time now, much closer to F=8's own ~17.5%
# resolution, though not identical -- an exact ratio match would need
# dt_snap~2.2; 1.0 is the specific value requested). trajectory_time
# increased 200->1000 (keeps ~1000 snapshots/trajectory despite the
# coarser dt_snap, so the K-step curriculum still has as many usable
# windows as before) -- spinup_time unchanged (200, already generous
# relative to this system's own timescale).
#
# Same encoder/propagator architecture as Section 196 (local_field
# n_sites=8/channels=2 -> d_latent=16, dense MLP propagator, markovian,
# --full-propagator) -- ONLY dt_snap changes, isolating it as the sole
# variable versus 196.
#
# Diagnostic windows rescaled to cover the SAME physical/Lyapunov-time
# duration as 196's own checks (196 used k=2000 steps * dt_snap=0.1 = 200
# model-time-units ~ 16 Lyapunov times for rollout; n_steps=2000,
# qr_every=10, warmup=200 * dt_snap=0.1 for the Lyapunov measurement) --
# at dt_snap=1.0 (10x coarser), the SAME physical coverage now needs 10x
# fewer steps: rollout k=200, Lyapunov n_steps=200/qr_every=5/warmup=20.
#
# Verified via a real (--profile full, --epochs 3) dry run of this exact
# recipe before this launch: no NaN, reconstruction converges even faster
# than 196's own dt_snap=0.1 case (recon 0.062 after 3 epochs vs. 196's
# unrecorded-but-comparable early value) -- consistent with each training
# step now covering a genuinely more informative amount of dynamics.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section197_lorenz96_n64_f4p2_dtsnap1

echo "=== [1/2] Stage 1 ONLY: Lorenz-96 N=64 F=4.2 dt_snap=1.0, local_field encoder (n_sites=8, channels=2, d_latent=16), dense MLP propagator (markovian), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset artifacts/datasets/lorenz96_trajectories_n64_f4.2_dtsnap1.h5 --nx 64 \
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

echo "=== [2/2] Diagnostics: reconstruction quality, standalone + multi-IC rollout (k=200, ~16 Lyapunov times at dt_snap=1.0), real Lyapunov spectrum vs. true L96 N=64/F=4.2 reference (lambda1=0.080, n_positive=5/64, D_KY=11.3) ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$AUX')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('d_latent:', d_latent)

DATASET = 'artifacts/datasets/lorenz96_trajectories_n64_f4.2_dtsnap1.h5'
with h5py.File(DATASET,'r') as f:
    traj20 = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, N = traj20.shape
with torch.no_grad():
    z_all = ae.encode(traj20.reshape(n*T, N)).reshape(n, T, -1)
z0b = z_all[:,0,:]

with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=200)
finite = torch.isfinite(traj_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('multi-IC (20) rollout first non-finite step:', first_bad)
maxz_per_t = traj_roll.abs().amax(dim=(0,2))
for t in [0,5,10,30,60,100,150,199]:
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
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=200, qr_every=5,
    dt_snap=1.0, warmup_steps=20, seed=0, max_abs_state=1e3,
)
print('lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('D_KY: could not bracket --', e)
print('(true L96 N=64/F=4.2 reference: lambda1=0.080, n_positive=5/64, D_KY=11.3)')
print('(Section 196, dt_snap=0.1, comparison: lambda1=0.056, n_positive=3/16, D_KY=4.93)')
" > artifacts/logs/lyapunov_${TAG}.log 2>&1
cat artifacts/logs/lyapunov_${TAG}.log

echo "=== Section 197 complete ==="
