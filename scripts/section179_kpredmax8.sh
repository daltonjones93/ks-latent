#!/bin/zsh
# User-directed 2026-09-16. Section 178 (mean-zero + bounded forcing on
# top of forced_burgers + Sakaguchi kernel) fixed the unbounded-divergence
# failure mode (176/177) -- standalone rollout stayed bounded (~1.8-2.5)
# for the full 200-step Stage-1-only checkpoint. But `visualize_rollout.py`
# revealed a DIFFERENT, more specific failure: the physical/latent
# Hovmoller plots show the model settling into a handful of nearly
# STATIONARY vertical bands after a short transient (~t<20) -- real
# spatial structure, but essentially no further TIME evolution within a
# single trajectory (relative RMSE oscillating 1.0-1.6, never tracking
# truth). The kernel amplitude A still never activated (0/49 unstable
# modes) even in this bounded regime.
#
# Root-cause hypothesis, user-directed: Stage 1's own `L_pred` term
# (`w_pred=0.5` by default, ACTIVE the entire 169-178 arc) has had its
# rollout length `k_pred_max` PINNED AT ITS DEFAULT OF 0 THE WHOLE TIME --
# confirmed directly (--k-pred-max was never passed in any of 169-178's
# Stage-1 commands, and k_pred_max=0 falls back to a fixed, never-ramped
# k_pred=2). This means Stage 1's own loss has NEVER had any signal about
# what the propagator does beyond a 2-step lookahead -- a propagator that
# nails 2 steps but freezes/collapses by step 10 pays ZERO cost in Stage 1,
# and only gets exposed (if at all) once Stage 2's much harder k-curriculum
# begins, by which point the map may already be locked into its cheap
# 2-step-optimal shape. This is the EXACT mechanism flagged in this
# project's own `docs/PROJECT_HANDOFF.md`-era notes ("L_pred's rollout
# K_pred=2 too short to generate a strong non-degenerate gradient from a
# collapsed starting point -- try a longer rollout") but never actually
# tested for this propagator family (spectral_pde_raw/forced_burgers).
#
# THIS section turns it on: NEW `--k-pred-max 8` (existing, previously-
# unused CLI flag -- ramps k_pred 2->8 over the first 30%% of Stage-1
# epochs, the SAME established default/curriculum shape already
# documented for `--multistep`, ported from the reference project's own
# benchmark). `--full-propagator` already sizes the propagator at
# hidden=128/n_blocks=3 regardless of `--multistep` (verified directly in
# the code), so `--multistep` itself is NOT needed alongside this --
# `--k-pred-max` alone is independent and sufficient.
#
# Verified via a real dry run (10 epochs) before this launch: k ramps
# 2->8 cleanly, cost stays cheap (7.7s->9.3s/epoch, nowhere near a 4x
# blowup), no nan, and the resulting (tiny-scale) checkpoint's standalone
# rollout stays bounded (grows 2.2->9.4, settles, no divergence).
#
# Kept identical to 178 otherwise, for direct comparison: forced_burgers
# + Sakaguchi kernel instability + mean-zero/bounded forcing + delta_cap,
# local_field encoder, spectral_pde_raw backbone, K=49/L=96.0, euler
# integrator, ode_substeps=1, same k_max=16/300-epoch Stage-2 curriculum.
# Also fixes a real gap noticed the same day: Stage 2 never had
# --checkpoint-every wired into ITS OWN launch command in any prior
# section script (the mechanism already existed in train_stage2_patched.py
# /loops.py, just never invoked) -- added here so a future kill mid-Stage-2
# doesn't lose all progress.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section179_kpredmax8

echo "=== [1/6] Stage 1: local_field encoder + spectral_pde_raw propagator, forced_burgers + Sakaguchi kernel instability + mean-zero/bounded forcing + --prop-delta-cap 0.5, --ode-substeps 1 + NEW --k-pred-max 8 (ramps L_pred's rollout 2->8 over first 30% of epochs -- previously pinned at k_pred=2 the ENTIRE 169-178 arc), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 \
  --spectral-field-kind forced_burgers --spectral-burgers-nu-init 1.0 --spectral-burgers-beta-max 1.0 \
  --spectral-burgers-kernel-instability --spectral-burgers-kernel-A-max 1.0 --spectral-burgers-kernel-width-init 1.0 \
  --spectral-burgers-forcing-max 1.0 \
  --spectral-integrator euler \
  --stable-linear-lr-factor 0.02 --prop-delta-cap 0.5 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.12 --spatial-signed --w-logdet 0.01 --w-smooth 0.012 \
  --w-channel-mean 0.01 \
  --full-propagator --amp \
  --k-pred-max 8 \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/6] Stage 2: spectral_pde_raw propagator continues (architecture, incl. mean-zero/bounded forcing + kernel-instability + delta_cap, inherited from the Stage-1 checkpoint), same --stable-linear-lr-factor 0.02, k_max=16 (Section 37's curriculum), 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k16_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --stable-linear-lr-factor 0.02 \
  --amp \
  --epochs 300 --k-max 16 --k-warmup-epochs 210 --k-mid 10 --k-mid-epochs 175 \
  --checkpoint-every 20 \
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
    n_unstable = int((Lhat_k > 0).sum())
    print('[$LABEL] number of modes with Lhat(k) > 0 (genuinely unstable):', n_unstable, 'out of', k.numel())

# Report the actual forcing's spatial mean/magnitude on real data -- did
# the zero-mean guarantee hold at scale, and is it still exercising real
# (nonzero) forcing rather than collapsing to zero output entirely?
derivs = __import__('ks_latent.models.spectral_field', fromlist=['synthesize_derivatives']).synthesize_derivatives(z_all[:, 0, :].float(), body.K, body.N_w, body.L, body.max_order) if hasattr(body, 'input_proj') else None
if derivs is not None:
    h = body.input_proj(derivs)
    for block in body.blocks:
        h = block(h)
    h = body.final_ln(h)
    raw_forcing = body.output_proj(h).squeeze(-1)
    capped = body.burgers_forcing_max * torch.tanh(raw_forcing / body.burgers_forcing_max)
    forcing = capped - capped.mean(dim=-1, keepdim=True)
    print('[$LABEL] forcing spatial mean (should be ~0): %.2e' % forcing.mean(dim=-1).abs().max().item())
    print('[$LABEL] forcing magnitude: mean_abs=%.4f  max_abs=%.4f' % (forcing.abs().mean().item(), forcing.abs().max().item()))
"
}

echo "=== [3/6] FULL Jacobian spectrum + conditioning check + fitted parameters + forcing stats on the FINAL propagator ==="
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

echo "=== Section 179 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
