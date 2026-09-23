#!/bin/zsh
# User-directed 2026-09-14. Sections 175/176 (forced Burgers with a REAL,
# stabilizing-only diffusion term -- constant nu or state-dependent
# nu(x)) both failed: 175 collapsed harder than any prior section
# (D_KY=0, lambda1=-0.269 -- 5-6x MORE contractive than 169-173's own
# ~-0.04 to -0.05), 176 diverged unboundedly (killed early, standalone
# max|z| growing 2.5->51 over 100 steps). Root cause diagnosed for 175:
# fitted nu (0.994) and beta (-0.006) BOTH barely moved from their safe,
# stabilizing init values -- the propagator never learned ANY genuine
# destabilizing mechanism, so it just damped everything to a fixed
# point. Nothing in the forced_burgers family up to this point provided
# a source of instability at all.
#
# User asked to read two papers and consider how to use them:
# arXiv:2609.01424 (Raj & Paul, coupled-map lattices with extended
# spatial coupling -- mostly a diagnostic lens: lambda_k ~ lambda_1 +
# ln|Lambda_k| relates the Lyapunov spectrum to the linear coupling
# operator's own eigenvalues, and their own finding that TOO MUCH
# coupling extent kills chaos outright is a good complementary
# experiment to try separately, not a mechanism to implement) and
# Sakaguchi (2000), "A Simple Model for Spatio-Temporal Chaos in an
# Unstable Burgers Equation," Prog. Theor. Phys. 103, 703 -- THIS is the
# one implemented here.
#
# Sakaguchi's own unstable Burgers equation:
#   w_t = integral(g(x-x') w_xx(x') dx') + w*w_x + nu*w_xx
# is diagonal in Fourier space: dw_hat_k/dt = -[g(k)+nu]*k^2*w_hat_k.
# Their OWN stability requirement: g(k)+nu < 0 for small k (genuine
# instability -- exactly the missing ingredient in 175/176) while g(k)
# -> 0 RAPIDLY for large k (so nu alone dominates and damps high
# wavenumbers -- unlike a literal k^4 term, whose magnitude GROWS
# without bound and forced Section 174's ode_substeps~64/graph-depth
# cost blowup). Their own numerical example used a Gaussian bump,
# g(k)=0.4*exp(-16k^2).
#
# New `--spectral-burgers-kernel-instability` (ks_latent/models/
# propagator.py, ks_latent/config.py) REPLACES forced_burgers' diffusion
# term with a genuinely diagonal Fourier multiplier applied directly to
# the K-mode spectral representation `field()` already receives (no
# extra transform needed):
#   Lhat(k) = -(A*exp(-k^2/width^2) + nu) * k^2
#   A = kernel_A_max*tanh(raw_A)   -- learnable, bounded to +-A_max, ANY
#                                     sign, starts at EXACTLY 0 (the
#                                     missing destabilizing ingredient)
#   width = softplus(raw_width)    -- learnable, always > 0
#   nu = softplus(raw_nu)          -- learnable, always > 0, SAME
#                                     mechanism as 175's constant-nu case
# At A=0 (init), Lhat(k) = -nu*k^2 EXACTLY -- identical to 175's own
# starting point, so this is a strict generalization, not a different
# start. Lhat(0)=0 identically regardless of A/width/nu (the overall k^2
# prefactor vanishes at DC) -- the mean/DC mode is architecturally
# untouched, matching true KS's own exact mass conservation for free.
#
# Verified directly before this launch (see conversation): at init, A=0
# recovers -nu*k^2 exactly; Lhat(0)=0 exactly; gradients flow correctly
# (raw_A's own gradient nonzero at init, bootstrapping A away from 0;
# width's gradient correctly zero AT A=0, the same cold-start pattern
# already used elsewhere in this codebase, becoming live once A moves);
# A stays bounded under an extreme-weight stress test; max|Lhat| at init
# is ~9.87 (~10x smaller than Section 174's own ~97-105 k^4-term
# magnitude), confirmed cheap via a real dry run -- clean gradual ramp
# k=2->16, no nan, ~5-23s/epoch (same scale as 175's own constant-nu
# timing, NOT 174's catastrophic ode_substeps=64 cost).
#
# Everything else identical to 175/176 for direct comparison: local_field
# encoder, spectral_pde_raw backbone, K=49/L=96.0, euler integrator,
# ode_substeps=1, --prop-delta-cap 0.5 (kept as a complementary
# architectural safeguard, same as before), same k_max=16/300-epoch
# Stage-2 curriculum.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section177_sakaguchi_kernel

echo "=== [1/6] Stage 1: local_field encoder + spectral_pde_raw propagator, forced_burgers with NEW --spectral-burgers-kernel-instability (Sakaguchi-style diagonal Fourier multiplier Lhat(k)=-(A*exp(-k^2/width^2)+nu)*k^2, A learnable+bounded+starts-at-0 supplies genuine low-k instability, nu guarantees high-k damping) + --prop-delta-cap 0.5, --ode-substeps 1, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 \
  --spectral-field-kind forced_burgers --spectral-burgers-nu-init 1.0 --spectral-burgers-beta-max 1.0 \
  --spectral-burgers-kernel-instability --spectral-burgers-kernel-A-max 1.0 --spectral-burgers-kernel-width-init 1.0 \
  --spectral-integrator euler \
  --stable-linear-lr-factor 0.02 --prop-delta-cap 0.5 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.12 --spatial-signed --w-logdet 0.01 --w-smooth 0.012 \
  --w-channel-mean 0.01 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/6] Stage 2: spectral_pde_raw propagator continues (architecture, incl. the kernel-instability mechanism/delta_cap, inherited from the Stage-1 checkpoint), same --stable-linear-lr-factor 0.02, k_max=16 (Section 37's curriculum), 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k16_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --stable-linear-lr-factor 0.02 \
  --amp \
  --epochs 300 --k-max 16 --k-warmup-epochs 210 --k-mid 10 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

_spectrum_check() {
  local PROP_PATH=$1
  local LABEL=$2
  mamba run -n da_env python -c "
import h5py, numpy as np, torch
import torch.nn.functional as F
from torch.func import jacrev
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.diagnostics import propagator_step_jacobian_spectral_norms

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$PROP_PATH')
ae.eval(); prop.eval()

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj.shape
with torch.no_grad():
    z_all = ae.encode(traj.reshape(n*T, NX)).reshape(n, T, -1)

res = propagator_step_jacobian_spectral_norms(prop, z_all, n_samples=200, seed=0)
print('[$LABEL] top singular value: median=%.4f  p95=%.4f  min=%.4f  max=%.4f' % (
    np.median(res), np.percentile(res, 95), res.min(), res.max()
))

z0 = z_all[0, 0]
J = jacrev(lambda z: prop.step_one(z.unsqueeze(0)).squeeze(0))(z0)
sv = torch.linalg.svdvals(J)
print('[$LABEL] singular values >= 1.0:', int((sv >= 1.0).sum()), 'out of', sv.numel())
print('[$LABEL] per-step volume-change factor:', sv.prod().item())

z0b = z_all[:, 0, :]
with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=60)
sep_start = (traj_roll[:, 5, :] - traj_roll[:, 0, :]).norm(dim=-1).mean().item()
sep_end = (traj_roll[:, -1, :] - traj_roll[:, -6, :]).norm(dim=-1).mean().item()
cross_sample_std_end = traj_roll[:, -1, :].std(dim=0).mean().item()
print('[$LABEL] rollout separation steps 0-5: %.4f   steps 54-59: %.4f   cross-sample z.std final step: %.4f' % (sep_start, sep_end, cross_sample_std_end))
print('[$LABEL] max|z| at final step:', traj_roll[:, -1, :].abs().max().item())

z_np = z_all.reshape(-1, z_all.shape[-1]).numpy()
cov = np.cov(z_np, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print('[$LABEL] spectrum: min_eig=%.4e  cond#=%.4e  top_eig=%.3f' % (eig[-1], eig[0]/eig[-1], eig[0]))

# Report fitted A/width/nu/beta directly -- did the model actually learn
# a genuine destabilizing A != 0, or did it drift back toward pure damping?
body = prop.body.inner if hasattr(prop.body, 'inner') else prop.body
if hasattr(body, 'raw_beta'):
    beta = (body.burgers_beta_max * torch.tanh(body.raw_beta)).item()
    print('[$LABEL] fitted beta: %.6f (bound +-%.2f)' % (beta, body.burgers_beta_max))
if getattr(body, 'burgers_kernel_instability', False):
    A = (body.burgers_kernel_A_max * torch.tanh(body.raw_kernel_A)).item()
    width = F.softplus(body.raw_kernel_width).item()
    nu = F.softplus(body.raw_nu).item()
    print('[$LABEL] fitted kernel: A=%.6f (bound +-%.2f)  width=%.6f  nu=%.6f' % (A, body.burgers_kernel_A_max, width, nu))
    k = body._kernel_k
    Lhat_k = -(A*torch.exp(-(k**2)/(width**2)) + nu) * (k**2)
    print('[$LABEL] Lhat(k) range: min=%.4f (k=%.3f)  max=%.4f (k=%.3f)  at k=0: %.6f' % (
        Lhat_k.min().item(), k[Lhat_k.argmin()].item(),
        Lhat_k.max().item(), k[Lhat_k.argmax()].item(), Lhat_k[0].item()
    ))
    n_unstable = int((Lhat_k > 0).sum())
    print('[$LABEL] number of modes with Lhat(k) > 0 (genuinely unstable):', n_unstable, 'out of', k.numel())
"
}

echo "=== [3/6] FULL Jacobian spectrum + conditioning check + fitted kernel parameters on the FINAL propagator ==="
_spectrum_check "$STAGE2_PROP" "stage2-final" > artifacts/logs/jacobiancheck_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_${STAGE2_TAG}.log

echo "=== [4/6] Visualization + Gate 3/4 (Lyapunov/D_KY, DA skill, D1-D9) ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --rollout-steps 200 --tag "$STAGE2_TAG" \
  > artifacts/logs/visualize_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/visualize_${STAGE2_TAG}.log

mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1

echo "=== [5/6] Standalone free-running divergence check ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$STAGE2_PROP')
ae.eval(); prop.eval()
with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][0], dtype=torch.float32)
u0 = traj[0].unsqueeze(0)
with torch.no_grad():
    z0 = ae.encode(u0)
    z_roll = prop.rollout(z0, z0, 200)
finite = torch.isfinite(z_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('first non-finite step (out of 200, -1 = never):', first_bad)
for t in [0,10,20,30,40,60,80,100,150,199]:
    v = z_roll[0, t]
    print(t, v.abs().max().item() if torch.isfinite(v).all() else 'nan/inf')
" > artifacts/logs/standalone_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/standalone_${STAGE2_TAG}.log

echo "=== [6/6] Self-spectrum mixing check across the standalone rollout, ACROSS MULTIPLE real ICs ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.spectral_field import encode_to_spectrum

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$STAGE2_PROP')
ae.eval(); prop.eval()
K = prop_cfg.spectral_K
with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    trajs = torch.tensor(f['trajectories'][:5], dtype=torch.float32)
for run_idx in range(5):
    for start in [0, 100]:
        u0 = trajs[run_idx, start].unsqueeze(0)
        with torch.no_grad():
            z0 = ae.encode(u0)
            z_roll = prop.rollout(z0, z0, 55).squeeze(0)
        if not torch.isfinite(z_roll).all():
            print('run=%d start=%d: DIVERGED' % (run_idx, start)); continue
        z_hat0 = encode_to_spectrum(z_roll[0:1], K).squeeze(0)
        z_hat54 = encode_to_spectrum(z_roll[54:55], K).squeeze(0)
        e0 = (z_hat0[:K]**2+z_hat0[K:2*K]**2); e54 = (z_hat54[:K]**2+z_hat54[K:2*K]**2)
        top0 = torch.topk(e0,3); top54 = torch.topk(e54,3)
        print('run=%d start=%3d: t0 top3=%s (%.3f) -> t54 top3=%s (%.3f)' % (
            run_idx, start,
            top0.indices.tolist(), (top0.values.sum()/e0.sum()).item(),
            top54.indices.tolist(), (top54.values.sum()/e54.sum()).item(),
        ))
" > artifacts/logs/mixing_check_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/mixing_check_${STAGE2_TAG}.log

echo "=== Section 177 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
