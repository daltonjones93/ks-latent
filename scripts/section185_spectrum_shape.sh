#!/bin/zsh
# User-directed 2026-09-17/18. Section 184 (relative delta_cap, 0.05 =
# 5%%/step) failed catastrophically once trained the full 200 epochs:
# bounded through t~20, then EXPONENTIAL runaway (79.8 at t=40, 11298 at
# t=60, 2.45 BILLION at t=100, 1.4e16 at t=150, nan by t~167-199). This
# was the FIFTH consecutive variant (180-184: fixed vs. learnable A/beta,
# hyperviscosity, forcing removed, unstable-mode floor, relative cap)
# landing on the same underlying failure: whatever cap shape is used,
# the model finds a configuration that saturates it every step in a
# consistent direction (linear runaway for an absolute cap, exponential
# for a relative one). User: "doesn't seem like 184 worked. any other
# ideas?" -> pivoted to fixing the TRAINING OBJECTIVE directly instead of
# another propagator-architecture tweak: "yep do 1" (add a genuine
# statistics/spectrum-matching loss term targeting real chaos, active
# throughout training, rather than hoping the propagator's own physical
# structure produces it as a side effect).
#
# THIS section discovered and reuses an EXISTING, already-validated
# mechanism instead of building a new one from scratch:
# `propagator_spectrum_shape_loss` (ks_latent/training/losses.py, added
# Section 131, "mode=markovian only, any backbone" -- never tried on the
# forced_burgers/kernel-instability family this whole 169-184 arc, since
# it predates it and was built for a different propagator backbone).
# Directly floors the propagator's own step-Jacobian singular-value
# spectrum: the top `n_expand` singular values toward an EXPANSIVE
# target (genuine chaos), the rest toward a lower CONTRACTING floor
# (KS is dissipative -- most directions should still contract, but not
# collapse to ~0). Computed once per epoch (cheap, small sample of 32),
# fully differentiable, added directly to the training loss --
# structurally different from every 180-184 attempt: this shapes the
# LOCAL LINEARIZATION's own spectral structure directly (the actual
# quantity that determines whether the map is chaotic), not the
# propagator's physical term coefficients or its per-step cap.
#
# Targets NOT guessed -- reused Section 133's own retargeting, computed
# directly from this project's best-trusted L=100 checkpoint's REAL
# Benettin/QR Lyapunov spectrum (Section 85, D_KY=21.76, n_positive=13,
# lambda1=0.086) converted to one-step multipliers exp(lambda_i*dt_snap):
# top 5 multipliers ~[1.09, 1.086, 1.085, 1.068, 1.066], bottom 5
# ~[0.631, 0.630, 0.623, 0.619, 0.604], count>=1.0: 13/44. NEW
# --w-spectrum-shape 0.4 --spectrum-shape-n-expand 13
# --spectrum-shape-expand-target 1.1 --spectrum-shape-contract-floor 0.6
# --spectrum-shape-n-samples 32 -- n_expand=13 independently matches this
# section's own kernel_unstable_target_modes=13 (kept from 183).
#
# Everything else identical to 183 (the LAST stable variant, before
# 184's relative-cap experiment): kernel A/beta learnable (init
# -2.0/-1.0), hyperviscosity mu_init=0.01, no forcing MLP,
# kernel-unstable floor (>=13 modes), k_pred_max=10, ABSOLUTE delta_cap
# 0.5 (NOT 184's relative cap -- reverted to the known-stable form),
# local_field encoder recipe.
#
# Verified via a real dry run (8 epochs) before this launch: no nan, no
# meaningful cost increase (once-per-epoch Jacobian computation, ~6.4s/
# epoch, unchanged from 183's own timing), standalone rollout bounded
# (settling ~3.3-3.6 across 100 steps) -- though 183's own 8-epoch dry
# run looked similarly fine before diverging with full training, so this
# is not conclusive on its own; the real 200-epoch run is the actual test.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section185_spectrum_shape

echo "=== [1/2] Stage 1 ONLY: local_field encoder + spectral_pde_raw propagator, forced_burgers + kernel A/beta learnable (init -2.0/-1.0) + kernel-unstable floor + hyperviscosity + no forcing MLP (same as 183) + NEW --w-spectrum-shape 0.4 (directly shapes the propagator's own step-Jacobian singular-value spectrum toward Section 85's real, measured Lyapunov-derived targets -- n_expand=13/expand_target=1.1/contract_floor=0.6) + --prop-delta-cap 0.5 (absolute, reverted from 184's broken relative cap) + --k-pred-max 10, --ode-substeps 1, --amp, 200 epochs ==="
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
  --w-spectrum-shape 0.4 --spectrum-shape-n-expand 13 --spectrum-shape-expand-target 1.1 --spectrum-shape-contract-floor 0.6 --spectrum-shape-n-samples 32 \
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

echo "=== Section 185 Stage 1 complete ==="
