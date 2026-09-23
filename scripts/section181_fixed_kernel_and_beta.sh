#!/bin/zsh
# User-directed 2026-09-16. Section 180 (kernel A FIXED at -2.0, not
# learnable -- the clean ablation testing whether A ever activating
# would change anything) confirmed the destabilizing mechanism WAS
# engaging (broke free of 178/179's frozen-at-small-scale collapse,
# 13/49 modes genuinely unstable throughout) but did NOT produce
# physically-scaled chaos: standalone rollout kept growing (max|z|
# 1.9->63.5 over 200 steps, no nan but no plateau at the true scale
# either), RMSE climbed straight through sqrt(2) saturation to ~6-7 (the
# model's own amplitude grew to several times the true attractor's
# scale), and the Hovmoller plots showed the model settling into MANY
# more (but still largely frozen, now saturated-at-extremes) bands
# rather than genuine bounded chaos. User: "if we remove delta_cap what
# would happen?"
#
# Diagnosis, given directly to the user: removing delta_cap would very
# likely make things WORSE (faster/harder blowup), because delta_cap was
# the ONLY thing bounding growth once A was forced unstable -- what's
# actually missing is the genuine PHYSICAL saturation mechanism real
# KS/Burgers turbulence relies on: the nonlinear ADVECTION term (beta*w*w_x)
# cascading energy from unstable low-k modes to damped high-k modes. beta
# had ALSO stayed near 0 in every one of 175-180's own fitted values --
# so with only A forced unstable, delta_cap (a fixed ABSOLUTE per-step
# cap, not scaled to the state's own magnitude) was doing 100%% of the
# bounding work artificially, rather than genuine nonlinear saturation
# doing it physically. User: "sure try that".
#
# THIS section adds the analogous ablation for beta: NEW
# --spectral-burgers-beta-fixed -1.0 (matches true KS's own advection
# coefficient) -- FIXES beta too, not learnable, alongside kernel A
# FIXED at -2.0 (same as 180). Tests whether real energy-cascade
# saturation emerges once BOTH the destabilizing (A) and the
# saturating-nonlinearity (beta) mechanisms are simultaneously forced
# active, rather than delta_cap alone doing all the (artificial)
# bounding work as in 180.
#
# Stage 1 ONLY again (same user direction as 180), same k_pred_max=8,
# same delta_cap/mean-zero-forcing/local_field encoder recipe as
# 179/180. Diagnostics run directly on the Stage-1 checkpoint.
#
# Verified via a real dry run (8 epochs) before this launch: no nan;
# growth is exactly LINEAR at delta_cap's own max rate (0.5/step) from
# ~t=20 onward -- confirms the mechanism is fully engaged this early,
# same qualitative signature as 180's own early dry run.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section181_fixed_kernel_and_beta

echo "=== [1/2] Stage 1 ONLY: local_field encoder + spectral_pde_raw propagator, forced_burgers + kernel A FIXED at -2.0 + NEW --spectral-burgers-beta-fixed -1.0 (beta ALSO forced, not learnable -- matches true KS's own advection coefficient) + mean-zero/bounded forcing + --prop-delta-cap 0.5 + --k-pred-max 8 (same as 179/180), --ode-substeps 1, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 \
  --spectral-field-kind forced_burgers --spectral-burgers-nu-init 1.0 --spectral-burgers-beta-max 1.0 \
  --spectral-burgers-kernel-instability --spectral-burgers-kernel-A-max 1.0 --spectral-burgers-kernel-width-init 1.0 \
  --spectral-burgers-kernel-A-fixed -2.0 --spectral-burgers-beta-fixed -1.0 \
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

beta = body._beta_fixed.item()
A = body._kernel_A_fixed.item()
width = F.softplus(body.raw_kernel_width).item()
nu = F.softplus(body.raw_nu).item()
print('FINAL Stage-1 checkpoint (epoch 199/200, A FIXED at -2.0, beta FIXED at -1.0):')
print('beta (FIXED, not learned):', beta)
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

echo "=== Section 181 Stage 1 complete ==="
