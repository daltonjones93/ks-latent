#!/bin/zsh
# User-directed 2026-09-16. Sections 178/179 (forced_burgers + Sakaguchi
# kernel instability + mean-zero/bounded forcing, with and without
# --k-pred-max 8) both fixed the unbounded-divergence problem but
# revealed a DIFFERENT failure: the model settles into a handful of
# nearly-stationary spatial/latent bands after a short transient --
# real spatial structure, no genuine sustained temporal chaos. In BOTH
# checkpoints the kernel amplitude A had drifted slightly NEGATIVE under
# gradient descent (178: A=-0.116; 179: A=-0.151) -- i.e. the local
# k-step MSE gradient was actively pushing TOWARD more damping, never
# discovering/using the destabilizing capacity the kernel mechanism
# offers. User: "is there any way to get kernel A to activate in 178 and
# 179? initialize differently? any ideas?"
#
# THIS section runs the clean causal ablation instead of a warm-start
# (which would likely just get walked back to ~0 slowly, same as before,
# under the same 178/179 evidence): NEW --spectral-burgers-kernel-A-fixed
# -2.0 architecturally FIXES A at this value -- not learnable at all (a
# buffer, no tanh reparametrization, no gradient) -- forcing a genuine
# low-k unstable band regardless of what gradient descent would have
# chosen. Verified directly: with nu~1.0/width~1.0 (their own init
# values), A=-2.0 gives Lhat(k)>0 for the lowest 12/49 kept modes.
# nu/width/beta STAY learnable (unchanged from 178/179) -- this isolates
# whether the FORCING MLP simply compensates for a forced-unstable A
# (reproducing the same frozen-band collapse -- meaning the kernel was
# never the actual bottleneck) or whether real, sustained dynamics
# emerge once the model can no longer avoid a destabilizing linear term.
#
# User-directed: "just run stage 1 and we can check what it looks like.
# keep everything else the same from 179" -- Stage 1 ONLY (no Stage 2),
# same k_pred_max=8, same delta_cap/mean-zero-forcing/local_field
# encoder recipe as 179. Diagnostics run directly on the Stage-1
# checkpoint (fitted parameters, standalone rollout, Jacobian, multi-IC
# separation, visualize_rollout plots) -- the same treatment used to
# diagnose 178/179's own Stage-1 checkpoints.
#
# Verified via a real dry run (8 epochs) before this launch: no nan
# (--prop-delta-cap 0.5 caps growth to at most LINEAR in time, never
# exponential, even with A forced unstable from the very first step);
# standalone rollout grows faster than 178/179's own early dry runs
# (expected -- nu/width/forcing haven't had time to adapt to the
# now-forced instability yet) but stays finite throughout.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section180_fixed_kernel_A

echo "=== [1/2] Stage 1 ONLY: local_field encoder + spectral_pde_raw propagator, forced_burgers + NEW --spectral-burgers-kernel-A-fixed -2.0 (A forced genuinely unstable, NOT learnable -- 12/49 modes have Lhat(k)>0 at nu/width init values) + mean-zero/bounded forcing + --prop-delta-cap 0.5 + --k-pred-max 8 (same as 179), --ode-substeps 1, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 \
  --spectral-field-kind forced_burgers --spectral-burgers-nu-init 1.0 --spectral-burgers-beta-max 1.0 \
  --spectral-burgers-kernel-instability --spectral-burgers-kernel-A-max 1.0 --spectral-burgers-kernel-width-init 1.0 \
  --spectral-burgers-kernel-A-fixed -2.0 \
  --spectral-burgers-forcing-max 1.0 \
  --spectral-integrator euler \
  --stable-linear-lr-factor 0.02 --prop-delta-cap 0.5 \
  --k-pred-max 8 \
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
A = body._kernel_A_fixed.item()
width = F.softplus(body.raw_kernel_width).item()
nu = F.softplus(body.raw_nu).item()
print('FINAL Stage-1 checkpoint (epoch 199/200, A FIXED at -2.0):')
print('fitted beta:', beta)
print('kernel: A=%.6f (FIXED, not learned)  width=%.6f (learned)  nu=%.6f (learned)' % (A, width, nu))
k = body._kernel_k
Lhat_k = -(A*torch.exp(-(k**2)/(width**2)) + nu) * (k**2)
n_unstable = int((Lhat_k > 0).sum())
print('modes with Lhat(k)>0 (genuinely unstable) with LEARNED width/nu:', n_unstable, 'out of', k.numel())

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
z_hat = encode_to_spectrum(z0b, body.K)
derivs = synthesize_derivatives(z_hat, body.K, body.N_w, body.L, body.max_order)
h = body.input_proj(derivs)
for block in body.blocks:
    h = block(h)
h = body.final_ln(h)
raw_forcing = body.output_proj(h).squeeze(-1)
capped = body.burgers_forcing_max * torch.tanh(raw_forcing / body.burgers_forcing_max)
forcing = capped - capped.mean(dim=-1, keepdim=True)
print('forcing spatial mean (should be ~0): %.2e' % forcing.mean(dim=-1).abs().max().item())
print('forcing magnitude: mean_abs=%.4f max_abs=%.4f' % (forcing.abs().mean().item(), forcing.abs().max().item()))

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

echo "=== Section 180 Stage 1 complete ==="
