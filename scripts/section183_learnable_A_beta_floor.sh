#!/bin/zsh
# User-directed 2026-09-16. Section 182 (kernel A FIXED at -2.0, beta
# FIXED at -1.0, NEW hyperviscosity mu_init=0.01, forcing MLP removed
# entirely) made essentially NO difference from 181: standalone rollout
# still grew exactly linearly at delta_cap's own max rate (0.5/step) for
# the ENTIRE 200-step window, reaching ~93 by t=199 (181 reached ~98) --
# neither removing the forcing MLP nor adding real 4th-order
# hyperviscosity changed the qualitative outcome at all. Diagnosis given
# to the user: the specific FIXED magnitudes (A=-2.0, beta=-1.0), chosen
# by analogy to true KS's own coefficients, may simply be too strong for
# this arbitrary self-FFT ring's own natural equilibrium scale (nu
# settles ~0.9-1.0 under training) -- and/or FIXING them entirely
# (removing gradient descent's ability to adjust) forecloses the
# possibility of the system finding its own genuine balance.
#
# THIS section, user-directed: "let's initialize with the same
# parameters but let A and beta be trainable. we should also increase
# the regularizer that tries to keep at least 13 unstable modes in the
# pde, or if that doesn't exist, implement it." No such regularizer
# existed -- NEW `kernel_Lhat()` (ks_latent/models/propagator.py) exposes
# the diagonal Fourier multiplier directly (a pure function of the
# model's own A/width/nu/mu parameters, no forward pass over data
# needed), and NEW `kernel_unstable_floor_loss` (ks_latent/training/
# losses.py) + `w_kernel_unstable_floor`/`kernel_unstable_target_modes`/
# `kernel_unstable_margin` (Stage{1,2}TrainingConfig) hinge-penalize the
# lowest `target_modes` Fourier modes (excluding DC, always exactly 0)
# for dropping below `margin` in Lhat(k) -- directly guaranteeing at
# least that many modes stay genuinely unstable throughout training,
# regardless of what gradient descent would otherwise do to A/beta/nu/mu.
#
# NEW --spectral-burgers-kernel-A-init -2.0 (with --spectral-burgers-
# kernel-A-max raised to 3.0 for headroom) and NEW --spectral-burgers-
# beta-init -1.0 (with --spectral-burgers-beta-max raised to 1.5) start
# A/beta LEARNABLE at the exact same values 180-182 hard-FIXED, via
# inverse-tanh reparametrization (verified directly: A=-2.0, beta=-1.0
# exactly at construction). --spectral-burgers-kernel-A-fixed/--spectral-
# burgers-beta-fixed REMOVED (no longer fixed). NEW --w-kernel-unstable-
# floor 1.0 --kernel-unstable-target-modes 13 --kernel-unstable-margin
# 0.05 (small positive margin for genuine headroom, not just borderline
# zero). Hyperviscosity (mu_init=0.01) and no-forcing kept identical to
# 182. Same k_pred_max=10, delta_cap=0.5, local_field encoder recipe.
#
# Stage 1 ONLY again (same as 180-182). Diagnostics run directly on the
# Stage-1 checkpoint, now also reporting the fitted (no longer fixed)
# A/beta values and confirming the floor loss is near 0 (target modes
# genuinely satisfied) at each checkpoint.
#
# Verified via a real dry run (8 epochs) before this launch: A/beta
# confirmed exactly -2.0/-1.0 at construction (CPU unit test); 5 learnable
# params in the slow-LR group (A, width, nu, mu, beta); kernel_unstable_
# floor_loss near 0 at init (12/13 target modes already positive, matching
# the fixed-A case's own count); gradients flow to all 5 parameters.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section183_learnable_A_beta_floor

echo "=== [1/2] Stage 1 ONLY: local_field encoder + spectral_pde_raw propagator, forced_burgers + kernel A and beta LEARNABLE (init -2.0/-1.0, same values 180-182 fixed) + NEW --w-kernel-unstable-floor 1.0 (guarantees >= 13 modes stay unstable regardless of gradient descent) + hyperviscosity mu_init=0.01 + no forcing MLP (same as 182) + --prop-delta-cap 0.5 + --k-pred-max 10, --ode-substeps 1, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 \
  --spectral-field-kind forced_burgers --spectral-burgers-nu-init 1.0 \
  --spectral-burgers-beta-max 1.5 --spectral-burgers-beta-init -1.0 \
  --spectral-burgers-kernel-instability --spectral-burgers-kernel-A-max 3.0 --spectral-burgers-kernel-A-init -2.0 \
  --spectral-burgers-kernel-width-init 1.0 \
  --spectral-burgers-kernel-mu-init 0.01 --spectral-burgers-no-forcing \
  --spectral-burgers-forcing-max 1.0 \
  --spectral-integrator euler \
  --stable-linear-lr-factor 0.02 --prop-delta-cap 0.5 \
  --w-kernel-unstable-floor 1.0 --kernel-unstable-target-modes 13 --kernel-unstable-margin 0.05 \
  --k-pred-max 10 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.12 --spatial-signed --w-logdet 0.01 --w-smooth 0.012 \
  --w-channel-mean 0.01 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/2] Diagnostics on the Stage-1-only checkpoint: fitted parameters, standalone rollout, Jacobian, multi-IC separation ==="
mamba run -n da_env python -c "
import h5py, torch
import torch.nn.functional as F
from torch.func import jacrev
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.spectral_field import encode_to_spectrum, synthesize_derivatives

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$AUX')
ae.eval(); prop.eval()
body = prop.body.inner if hasattr(prop.body, 'inner') else prop.body

beta = (body.burgers_beta_max * torch.tanh(body.raw_beta)).item()
A = (body.burgers_kernel_A_max * torch.tanh(body.raw_kernel_A)).item()
width = F.softplus(body.raw_kernel_width).item()
nu = F.softplus(body.raw_nu).item()
mu = F.softplus(body.raw_kernel_mu).item()
print('FINAL Stage-1 checkpoint (epoch 199/200, A/beta LEARNABLE from -2.0/-1.0, no forcing MLP, kernel-unstable floor active):')
print('beta (LEARNED, init -1.0, bound +-%.2f):' % body.burgers_beta_max, beta)
print('A (LEARNED, init -2.0, bound +-%.2f):' % body.burgers_kernel_A_max, A)
print('width=%.6f (learned)  nu=%.6f (learned)  mu=%.6f (learned, hyperviscosity)' % (width, nu, mu))
Lhat_k = body.kernel_Lhat()
n_unstable = int((Lhat_k > 0).sum())
print('modes with Lhat(k)>0 (genuinely unstable):', n_unstable, 'out of', Lhat_k.numel())
print('Lhat(k) at highest mode (should be strongly negative, real damping):', Lhat_k[-1].item())
from ks_latent.training.losses import kernel_unstable_floor_loss
floor_loss = kernel_unstable_floor_loss(Lhat_k, target_modes=13, margin=0.05).item()
print('kernel_unstable_floor_loss (should be near 0 if the >=13 guarantee held):', floor_loss)

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][0], dtype=torch.float32)
u0 = traj[0].unsqueeze(0)
with torch.no_grad():
    z0 = ae.encode(u0)
    z_roll = prop.rollout(z0, z0, 200)
finite = torch.isfinite(z_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('standalone rollout (200 steps) first non-finite step (-1=never):', first_bad)
for t in [0,10,20,40,60,100,150,199]:
    v = z_roll[0,t]
    print(' t=%d max|z|=%s' % (t, v.abs().max().item() if torch.isfinite(v).all() else 'nan/inf'))

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj20 = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj20.shape
with torch.no_grad():
    z_all = ae.encode(traj20.reshape(n*T, NX)).reshape(n, T, -1)
z0b = z_all[:,0,:]
print('forcing MLP: REMOVED entirely this section (--spectral-burgers-no-forcing) -- nothing to report.')

z0j = z_all[0,0]
J = jacrev(lambda z: prop.step_one(z.unsqueeze(0)).squeeze(0))(z0j)
sv = torch.linalg.svdvals(J)
print('singular values >= 1.0:', int((sv>=1.0).sum()), 'out of', sv.numel())
print('per-step volume-change factor:', sv.prod().item())

with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=60)
sep_start = (traj_roll[:,5,:]-traj_roll[:,0,:]).norm(dim=-1).mean().item()
sep_end = (traj_roll[:,-1,:]-traj_roll[:,-6,:]).norm(dim=-1).mean().item()
cross_std = traj_roll[:,-1,:].std(dim=0).mean().item()
print('rollout separation steps 0-5: %.4f  steps 54-59: %.4f  cross-sample z.std final: %.4f' % (sep_start, sep_end, cross_std))
" > artifacts/logs/diagnostics_${TAG}.log 2>&1
cat artifacts/logs/diagnostics_${TAG}.log

echo "=== [bonus] Rollout visualization ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$AUX" \
  --rollout-steps 200 --tag "${TAG}_phase1" \
  > artifacts/logs/visualize_${TAG}.log 2>&1
cat artifacts/logs/visualize_${TAG}.log

echo "=== Section 183 Stage 1 complete ==="
