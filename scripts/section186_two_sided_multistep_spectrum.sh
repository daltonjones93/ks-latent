#!/bin/zsh
# User-directed 2026-09-18. Section 185 added propagator_spectrum_shape_loss
# (w_spectrum_shape) on top of 183's recipe, targeting Section 85's own
# real Lyapunov-derived spectrum (n_expand=13, expand_target=1.1,
# contract_floor=0.6) -- but a direct measurement of the TRAINED
# checkpoint's own Jacobian singular values found the mechanism inert:
# top-15 = [4.65, 3.94, 3.72, 3.47, 3.46, 3.15, 3.05, 2.75, 1.85, 1.63,
# 1.58, 1.31, 1.25, 1.16, 1.16] -- already 3-4x ABOVE expand_target=1.1,
# so the ONE-SIDED floor (relu(target - sv)^2) was already fully
# satisfied and contributed exactly zero gradient for the entire
# 200-epoch run. User: "I think we should try the two sided approach for
# 186. also is there a way to force the pde's singular vectors ... to go
# from expansive to contractive and back again? ... if the expansive
# singular vectors all feed into other expansive singular vectors in the
# evolution of the system, we will see runaway growth."
#
# THIS section adds TWO new mechanisms (ks_latent/training/losses.py):
#
# 1. `--spectrum-shape-two-sided`: propagator_spectrum_shape_loss now
#    uses a plain squared-error match ((sv-target)^2) in both groups
#    instead of a one-sided hinge -- exceeding the target now costs as
#    much as falling short, which is the only way this mechanism can pull
#    an already-excessive singular value back DOWN.
#
# 2. `--w-spectrum-shape-multistep 0.4` (NEW function
#    propagator_multistep_spectrum_shape_loss): directly answers the
#    singular-VECTOR question. Autodiff has no cheap differentiable
#    handle on singular vectors themselves, so this targets the same
#    failure mode indirectly but precisely -- it constrains the singular
#    VALUES of the COMPOSED k-step Jacobian d(z_{n+k})/d(z_n) (chaining
#    step_one k times before one Jacobian call), not the one-step
#    Jacobian. A map whose expansive directions persistently re-feed the
#    SAME subspace every step (no mixing into a contracting direction)
#    has its k-step top singular value compound as s^k (measured
#    one-step s~4.65, k=10 -> ~2.7e6); a map with genuine Oseledets-style
#    mixing between expansive/contractive directions instead converges
#    toward the REAL asymptotic Lyapunov rate exp(lambda_1*k*dt_snap) =
#    expand_target**k (~1.1^10~2.6, since expand_target was itself
#    derived as exp(lambda_1*dt_snap) from Section 85's real spectrum).
#    Penalizing (k-step top sv) - expand_target**k with a two-sided
#    squared error directly punishes persistent self-feeding using only
#    quantities autodiff can already compute. k=10 matches --k-pred-max.
#
# Everything else identical to 185 (kernel A/beta learnable init
# -2.0/-1.0, hyperviscosity mu_init=0.01, no forcing MLP,
# kernel-unstable floor >=13 modes, k_pred_max=10, ABSOLUTE delta_cap
# 0.5, local_field encoder recipe). Stage 1 ONLY, dry-run first.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section186_two_sided_multistep_spectrum

echo "=== [1/2] Stage 1 ONLY: same as 185 + --spectrum-shape-two-sided + NEW --w-spectrum-shape-multistep 0.4 (composed k=10-step Jacobian spectrum, two-sided, targets 1.1^10/0.6^10) ==="
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
  --w-spectrum-shape 0.4 --spectrum-shape-n-expand 13 --spectrum-shape-expand-target 1.1 --spectrum-shape-contract-floor 0.6 --spectrum-shape-n-samples 32 --spectrum-shape-two-sided \
  --w-spectrum-shape-multistep 0.4 --spectrum-shape-multistep-k 10 --spectrum-shape-multistep-n-samples 16 \
  --k-pred-max 10 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.12 --spatial-signed --w-logdet 0.01 --w-smooth 0.012 \
  --w-channel-mean 0.01 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/2] Diagnostics on the Stage-1-only checkpoint: fitted parameters, standalone rollout, Jacobian (one-step AND k=10-step composed), multi-IC separation ==="
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
print('FINAL Stage-1 checkpoint (epoch 199/200, two-sided spectrum-shape + multistep k=10):')
print('beta (LEARNED, init -1.0, bound +-%.2f):' % body.burgers_beta_max, beta)
print('A (LEARNED, init -2.0, bound +-%.2f):' % body.burgers_kernel_A_max, A)
print('width=%.6f (learned)  nu=%.6f (learned)  mu=%.6f (learned, hyperviscosity)' % (width, nu, mu))
Lhat_k = body.kernel_Lhat()
n_unstable = int((Lhat_k > 0).sum())
print('modes with Lhat(k)>0 (genuinely unstable):', n_unstable, 'out of', Lhat_k.numel())
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

z0j = z_all[0,0]
J1 = jacrev(lambda z: prop.step_one(z.unsqueeze(0)).squeeze(0))(z0j)
sv1 = torch.linalg.svdvals(J1)
print('ONE-STEP: singular values >= 1.0:', int((sv1>=1.0).sum()), 'out of', sv1.numel())
print('ONE-STEP top 15:', [round(x,3) for x in sv1[:15].tolist()])
print('ONE-STEP per-step volume-change factor:', sv1.prod().item())

def step10(z):
    out = z
    for _ in range(10):
        out = prop.step_one(out.unsqueeze(0)).squeeze(0)
    return out
J10 = jacrev(step10)(z0j)
sv10 = torch.linalg.svdvals(J10)
print('K=10-STEP COMPOSED: top 15 singular values:', [round(x,3) for x in sv10[:15].tolist()])
print('K=10-STEP COMPOSED: targets were 1.1^10=%.3f (top13) / 0.6^10=%.6f (rest)' % (1.1**10, 0.6**10))
print('K=10-STEP COMPOSED: bottom 10 singular values:', [round(x,6) for x in sv10[-10:].tolist()])

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

echo "=== Section 186 Stage 1 complete ==="
