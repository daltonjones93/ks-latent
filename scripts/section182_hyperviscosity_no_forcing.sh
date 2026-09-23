#!/bin/zsh
# User-directed 2026-09-16. Section 181 (kernel A FIXED at -2.0 AND beta
# FIXED at -1.0, matching true KS's own advection coefficient) made
# things WORSE than 180, not better: RMSE climbed almost linearly for
# the ENTIRE 200-step window to ~32 (never saturating anywhere, unlike
# 180's own plateau around 6-7.5), max|z| grew exactly linearly at
# delta_cap's own max rate (0.5/step) the whole time with zero
# deceleration, and the physical Hovmoller plot showed an even MORE
# extreme version of the frozen-at-saturated-extremes pattern (nearly
# the entire plot pinned to the colorbar edges past t~30).
#
# Diagnosis, given directly to the user: real KS-class saturation
# depends on the 4th-order hyperviscosity term (-w_xxxx, damping ~k^4)
# to absorb the energy the advection term cascades to high wavenumbers.
# The "forced_burgers" family (Section 175 onward) deliberately dropped
# this term for cost reasons (a k^4 term's magnitude grows unboundedly,
# forcing Section 174's ode_substeps~64) -- so even with the advection
# term now genuinely active, there was still no real high-k dissipation
# mechanism to absorb the cascaded energy; delta_cap alone was still
# doing 100%% of the bounding work. ALSO: the forcing MLP (magnitude
# ~0.7-1.0, not negligible even mean-zero) was still active throughout
# 178-181 and could itself be contributing to or masking the dynamics
# rather than the pure physical terms alone being incapable of bounded
# chaos. User: "let's keep the sakaguchi setup, add a hyperviscosity
# term and get rid of the forcing mlp term. keep everything else the
# same from 181, but extend rollout to k = 10."
#
# THIS section: NEW --spectral-burgers-kernel-mu-init 0.01 extends
# Lhat(k) with a genuine -mu*k^4 term (mu=softplus(raw), always > 0,
# matching true KS's own -w_xxxx exactly). NEW --spectral-burgers-no-forcing
# removes the forcing MLP entirely (no trunk built at all -- field()
# returns a plain zero for it), leaving ONLY the hardcoded/constrained
# physical terms (advection, diffusion, kernel instability,
# hyperviscosity), matching Sakaguchi's own equation (1) structurally --
# no separate learned correction term at all. --k-pred-max bumped 8->10
# per the user's direction. Kernel A FIXED at -2.0 and beta FIXED at
# -1.0 kept identical to 181.
#
# Stage 1 ONLY again (same as 180/181), same delta_cap/local_field
# encoder recipe otherwise. Diagnostics run directly on the Stage-1
# checkpoint.
#
# Verified via a real dry run (8 epochs) before this launch: no nan
# (mu_init=0.01 chosen as a modest, real hyperviscosity value -- note
# delta_cap already provides the actual numerical safety net regardless
# of any naive linear-stability estimate for this term); growth pattern
# at this early stage still linear at delta_cap's own max rate, same
# qualitative signature as 180/181's own early dry runs -- the real test
# is whether the FULL 200-epoch run behaves differently once mu/width/nu
# have had time to adapt and (without the forcing MLP fighting them) the
# physical terms alone can find a genuine balance.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section182_hyperviscosity_no_forcing

echo "=== [1/2] Stage 1 ONLY: local_field encoder + spectral_pde_raw propagator, forced_burgers + kernel A FIXED at -2.0 + beta FIXED at -1.0 + NEW --spectral-burgers-kernel-mu-init 0.01 (genuine 4th-order hyperviscosity, -mu*k^4) + NEW --spectral-burgers-no-forcing (forcing MLP removed entirely -- pure Sakaguchi-style physical terms only) + --prop-delta-cap 0.5 + --k-pred-max 10 (bumped from 8), --ode-substeps 1, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 \
  --spectral-field-kind forced_burgers --spectral-burgers-nu-init 1.0 --spectral-burgers-beta-max 1.0 \
  --spectral-burgers-kernel-instability --spectral-burgers-kernel-A-max 1.0 --spectral-burgers-kernel-width-init 1.0 \
  --spectral-burgers-kernel-A-fixed -2.0 --spectral-burgers-beta-fixed -1.0 \
  --spectral-burgers-kernel-mu-init 0.01 --spectral-burgers-no-forcing \
  --spectral-burgers-forcing-max 1.0 \
  --spectral-integrator euler \
  --stable-linear-lr-factor 0.02 --prop-delta-cap 0.5 \
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

beta = body._beta_fixed.item()
A = body._kernel_A_fixed.item()
width = F.softplus(body.raw_kernel_width).item()
nu = F.softplus(body.raw_nu).item()
mu = F.softplus(body.raw_kernel_mu).item()
print('FINAL Stage-1 checkpoint (epoch 199/200, A FIXED at -2.0, beta FIXED at -1.0, no forcing MLP):')
print('beta (FIXED, not learned):', beta)
print('kernel: A=%.6f (FIXED, not learned)  width=%.6f (learned)  nu=%.6f (learned)  mu=%.6f (learned, hyperviscosity)' % (A, width, nu, mu))
k = body._kernel_k
Lhat_k = -(A*torch.exp(-(k**2)/(width**2)) + nu) * (k**2) - mu*(k**4)
n_unstable = int((Lhat_k > 0).sum())
print('modes with Lhat(k)>0 (genuinely unstable) with LEARNED width/nu/mu:', n_unstable, 'out of', k.numel())
print('Lhat(k) at highest mode (should be strongly negative, real damping):', Lhat_k[-1].item())

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

echo "=== Section 182 Stage 1 complete ==="
