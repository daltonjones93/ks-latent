#!/bin/zsh
# User-directed 2026-09-14. Section 174 ("KS-shaped template" -- learnable
# but sign-guaranteed w_xx/w_xxxx, matching true KS's own k^2-k^4 linear
# operator) was numerically sound (verified: fitted physical coefficients
# exactly -1.0 at init, real gradient) but PROHIBITIVELY expensive: the
# stiff -w_xxxx term needs ode_substeps~64 for forward-Euler stability at
# this K=49/L=96 self-FFT ring, and graph depth (k*ode_substeps, up to
# 16*64=1024) scales far worse than linearly in backward-pass cost on
# this MPS setup -- measured 1061s/epoch STEADY STATE at k=16 (vs.
# 16.7s/epoch for Section 173's ode_substeps=1 baseline), projecting to
# DAYS for a full 300-epoch Stage 2 run. User: "I think the current
# approach is not worth testing for 10 hours."
#
# THIS section is the user's own revised proposal: forced Burgers with
# REAL (positive, unconditionally stabilizing) viscosity, where the
# forcing is a LEARNED function of the local latent state (not a fixed
# function of physical position, which Section 174's own discussion
# showed couldn't have sustained chaos):
#
#   w_t = beta*w*w_x + nu*w_xx + g_theta(w, w_x, ..., w^(max_order))
#
# New `spectral_field_kind="forced_burgers"` (ks_latent/models/
# propagator.py, ks_latent/config.py):
#   - nu = softplus(raw_nu), architecturally ALWAYS > 0 regardless of
#     raw_nu -- ordinary diffusion, Re(lambda(k)) = -nu*k^2 -> -inf as
#     k->inf UNCONDITIONALLY for ANY positive nu. Simpler/more robust than
#     true KS's own delicate k^2-k^4 cancellation, AND removes the
#     4th-order term entirely -- max |eigenvalue| drops from ~97
#     (Section 174's char_k^4) to ~10 (nu=1 * char_k^2), ~10x relaxation
#     of the forward-Euler stability constraint.
#   - g_theta: reuses the exact field_kind="mlp" pointwise-MLP
#     architecture, added as a residual forcing term (zero at init) on
#     top of the two hardcoded/constrained physical terms.
#
# FIRST version (advection hardcoded at exactly -1.0, nu_init=1.0,
# ode_substeps=1) STILL diverged to nan by epoch 4/k=6 under a real
# gradual k-ramp (verified directly, not just an artificial stress test)
# -- the nonlinear advection term has its OWN amplitude-dependent
# CFL-type explicit-Euler stability constraint, entirely independent of
# nu; lowering nu alone can never fix an advection-driven blowup.
# ode_substeps=8 alone (to relax that CFL constraint numerically) was
# tested and found STABLE but reintroduced the exact same catastrophic
# graph-depth cost problem (1061s/epoch class) that motivated leaving
# Section 174 in the first place -- confirms the practical constraint is
# k*ode_substeps graph depth in general, not which stiffness-reduction
# trick is used.
#
# TWO complementary fixes, both verified via direct dry runs (gradual
# k=2->16 ramp, ode_substeps=1, real supervised training, not just a
# forward-pass check) before this launch:
#   1. `--prop-delta-cap 0.5` (PropagatorConfig.delta_cap, pre-existing
#      mechanism -- architecturally bounds EVERY step's total update via
#      delta_cap*tanh(delta/delta_cap), regardless of what the internal
#      integrator computes). Verified ALONE to fully resolve the nan --
#      clean gradual ramp k=2->16, no nan anywhere, ~5-23s/epoch (same
#      scale as Section 173's own ode_substeps=1 baseline).
#   2. `beta = burgers_beta_max*tanh(raw_beta)` (NEW, user-directed same
#      day: "why don't we give -u*u_x a coefficient term too, if that's
#      the cause of the instability") -- a genuine learnable coefficient
#      on the advection term, architecturally bounded to
#      [-beta_max, +beta_max] regardless of training (same hard-cap
#      philosophy as delta_cap), raw_beta starting at exactly 0 (pure
#      diffusion+forcing at init, advection strength discovered gradually
#      via the same low-LR --stable-linear-lr-factor group nu uses).
#      Verified in combination with delta_cap: also clean, no nan,
#      same cost scale.
#
# Launched with BOTH active together (complementary safeguards, not
# redundant -- delta_cap bounds the OUTER step regardless of cause,
# beta bounds the SPECIFIC term suspected of driving the instability).
#
# Same field settings as Sections 167/169/174 otherwise (local_field
# encoder, spectral_pde_raw backbone, K=49/L=96.0, euler integrator,
# ode_substeps=1 -- NOT 64 like 174, this is the whole point), fresh
# Stage 1 (new mechanism, can't reuse a prior checkpoint), Stage 2 with
# the SAME k_max=16/300-epoch curriculum as every prior section in this
# saga for a direct comparison.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section175_forced_burgers

echo "=== [1/6] Stage 1: local_field encoder + spectral_pde_raw propagator, NEW --spectral-field-kind forced_burgers (w_t = beta*w*w_x + nu*w_xx + g_theta(...), nu>0 architectural via softplus, beta in [-1,1] architectural via tanh, both learnable with --stable-linear-lr-factor 0.02) + --prop-delta-cap 0.5 (architectural per-step bound), --ode-substeps 1 (cheap -- no stiff 4th-order term to sub-step against), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 \
  --spectral-field-kind forced_burgers --spectral-burgers-nu-init 1.0 --spectral-burgers-beta-max 1.0 \
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

echo "=== [2/6] Stage 2: spectral_pde_raw propagator continues (architecture, incl. forced_burgers/delta_cap, inherited from the Stage-1 checkpoint), same --stable-linear-lr-factor 0.02, k_max=16 (Section 37's curriculum), 300 epochs ==="
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

# Report the fitted nu/beta directly -- did they stay near their physical
# init values, or drift (while staying within their architectural bounds)?
body = prop.body.inner if hasattr(prop.body, 'inner') else prop.body
if hasattr(body, 'raw_nu'):
    import torch.nn.functional as F
    nu = F.softplus(body.raw_nu).item()
    beta = (body.burgers_beta_max * torch.tanh(body.raw_beta)).item()
    print('[$LABEL] fitted nu: %.6f   fitted beta: %.6f (bound +-%.2f)' % (nu, beta, body.burgers_beta_max))
"
}

echo "=== [3/6] FULL Jacobian spectrum + conditioning check + fitted nu/beta on the FINAL propagator ==="
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

echo "=== Section 175 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
