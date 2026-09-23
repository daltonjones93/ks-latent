#!/bin/zsh
# User-directed 2026-09-16/17. Sections 180-183 (four independent
# variants: fixed vs. learnable A/beta, hyperviscosity, forcing MLP
# removed, an unstable-mode-count regularizer) ALL converged to the
# identical failure signature -- sustained growth to a large, unphysical
# scale, never settling to bounded chaos at the correct scale (183:
# max|z| reaching ~89-93 by t=199, RMSE ~19-32, essentially unchanged
# across all four). User's own observation: "183 looks similar to 182."
#
# Diagnosis: the ONE thing common to all four variants, never varied,
# was `delta_cap` -- a FIXED ABSOLUTE per-step bound (0.5), not scaled to
# the state's own magnitude. Once real instability is present, growth
# continues until whatever scale makes that fixed 0.5 negligible in
# RELATIVE terms, rather than reaching a genuine dynamical balance. User:
# "fine implement that and test it in 184."
#
# THIS section: NEW `delta_cap_relative` (PropagatorConfig/
# AuxPropagatorConfig, ks_latent/models/propagator.py's `capped_delta`) --
# when set, the cap becomes `delta_cap * ||z_ref||` (the state's own
# per-sample norm) instead of a bare constant, so the allowed step size
# scales PROPORTIONALLY with the state's current magnitude. New CLI flag
# --prop-delta-cap-relative.
#
# CRITICAL finding during dry-run validation: --prop-delta-cap 0.5
# (the SAME numeric value used the whole 175-183 arc, now interpreted as
# a 50%%-of-norm-per-step fraction instead of an absolute magnitude)
# caused catastrophic EXPONENTIAL blowup to inf by step ~28 -- a relative
# cap compounds MULTIPLICATIVELY each step, unlike the absolute cap's
# additive/linear growth, so the same numeric value means something
# completely different in relative mode. Retried with --prop-delta-cap
# 0.05 (5%%/step, closer to true KS's own Lyapunov growth rate ~0.04-0.1)
# instead: STABLE. Standalone 200-step rollout stayed bounded in a narrow
# 3.0-3.4 range for the ENTIRE window (vs. 180-183's unbounded growth to
# 60-98) -- the best early signal in the whole 169-184 arc. Confirmed
# genuinely dynamic, not another frozen state: per-step change norm
# 4.3-6.1 (substantial relative to the ~3-3.4 amplitude), and multi-IC
# rollout separation stayed roughly CONSTANT (5.52->5.01 over 60 steps --
# neither collapsing toward 0 like every prior collapse, nor growing
# unboundedly like 180-183's own divergence).
#
# Kept identical to 183 otherwise: kernel A/beta learnable (init
# -2.0/-1.0), hyperviscosity mu_init=0.01, no forcing MLP,
# kernel-unstable floor (target >= 13 modes), k_pred_max=10, local_field
# encoder recipe.
#
# Stage 1 ONLY again (same as 180-183). Diagnostics run directly on the
# Stage-1 checkpoint.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section184_relative_delta_cap

echo "=== [1/2] Stage 1 ONLY: local_field encoder + spectral_pde_raw propagator, forced_burgers + kernel A/beta learnable (init -2.0/-1.0) + kernel-unstable floor + hyperviscosity + no forcing MLP (same as 183) + NEW --prop-delta-cap-relative (cap now scales with the state's own norm, 0.05 = 5%%/step, instead of a fixed absolute 0.5) + --k-pred-max 10, --ode-substeps 1, --amp, 200 epochs ==="
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
  --stable-linear-lr-factor 0.02 --prop-delta-cap 0.05 --prop-delta-cap-relative \
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

echo "=== Section 184 Stage 1 complete ==="
