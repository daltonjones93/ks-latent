#!/bin/zsh
# User-directed 2026-09-14. Section 175 (forced Burgers, CONSTANT nu/beta,
# ode_substeps=1, --prop-delta-cap 0.5) trained cleanly (no nan, both
# stages complete, ~50min total) but its diagnostics point at collapse
# again: per-step volume-change factor 3.96e-10 (even more contractive
# than Sections 169-173's own ~3.5e-4), val_kmax_mse plateaued ~0.51,
# final relative RMSE=1.0 at t=201 (no forecast skill). Notably DIFFERENT
# from 169-173's signature though: 44/96 Jacobian singular values are >=
# 1.0 (some directions locally EXPANDING, not uniformly damped) --
# genuinely anisotropic, not simple uniform contraction. Fitted nu=0.994,
# beta=-0.006 -- both barely moved from init (the slow-LR mechanism
# worked exactly as designed), meaning the trained propagator leaned
# almost entirely on real diffusion (nu~1) + the forcing MLP, with
# essentially zero nonlinear advection engaged.
#
# THIS section tests the follow-up sketched the same day, user-directed:
# "should we consider creating a pde that isn't a polynomial? it could be
# a more generic nonlinear function of the derivatives right?" --
# generalizes the CONSTANT nu to a state-dependent
# nu(w, w_x, w_xx, w_xxx, w_xxxx), computed by its own small MLP trunk
# (kept separate from the forcing MLP so it gets its own slow-LR
# treatment without throttling the forcing term's own learning speed).
# Architectural guarantee preserved exactly: nu(x) = softplus(trunk(x))
# stays positive AT EVERY POINT regardless of training -- the same
# unconditional diffusion guarantee the constant case has, just able to
# vary in strength with the local field shape instead of being one fixed
# number everywhere. Zero-init gives nu(x) == 1.0 exactly everywhere at
# construction, identical to Section 175's own starting point.
#
# New CLI flag --spectral-burgers-nonlinear-nu
# (ks_latent/models/propagator.py, ks_latent/config.py). Verified on CPU
# before this launch (see conversation): zero-init recovers the
# constant-nu case exactly; nu(x) stays strictly positive under a direct
# stress test (extreme trunk weights + extreme inputs); gradient reaches
# the trunk's output layer immediately (bootstraps on the first step,
# same cold-start pattern zero_init uses elsewhere in this codebase) --
# but this is the FIRST real training run of this mechanism, launched
# concurrently with Section 175's own (CPU-bound, Lyapunov/Benettin-QR)
# diagnostics still finishing, since training itself is MPS-bound and
# does not contend with that CPU work.
#
# Everything else identical to Section 175: local_field encoder,
# spectral_pde_raw backbone, K=49/L=96.0, euler integrator,
# ode_substeps=1, --prop-delta-cap 0.5 (kept as the same complementary
# safeguard), same k_max=16/300-epoch Stage-2 curriculum, for a direct
# comparison against 175's own numbers.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section176_forced_burgers_nonlinearnu

echo "=== [1/6] Stage 1: local_field encoder + spectral_pde_raw propagator, forced_burgers with NEW --spectral-burgers-nonlinear-nu (nu becomes nu(w,w_x,...,w_xxxx) via its own MLP trunk, softplus'd -- always positive, zero-init gives nu(x)==1.0 everywhere) + --prop-delta-cap 0.5, --ode-substeps 1, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 \
  --spectral-field-kind forced_burgers --spectral-burgers-nu-init 1.0 --spectral-burgers-beta-max 1.0 \
  --spectral-burgers-nonlinear-nu \
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

echo "=== [2/6] Stage 2: spectral_pde_raw propagator continues (architecture, incl. nonlinear-nu trunk/delta_cap, inherited from the Stage-1 checkpoint), same --stable-linear-lr-factor 0.02, k_max=16 (Section 37's curriculum), 300 epochs ==="
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

# Report fitted beta (scalar) and the RANGE of nu(x) across real data
# (nu is now a function, not a scalar) -- did it stay near 1.0 everywhere
# like at init, or did it learn real spatial/state variation?
body = prop.body.inner if hasattr(prop.body, 'inner') else prop.body
if hasattr(body, 'raw_beta'):
    beta = (body.burgers_beta_max * torch.tanh(body.raw_beta)).item()
    print('[$LABEL] fitted beta: %.6f (bound +-%.2f)' % (beta, body.burgers_beta_max))
if getattr(body, 'burgers_nonlinear_nu', False):
    from ks_latent.models.spectral_field import encode_to_spectrum, synthesize_derivatives
    z_hat_all = encode_to_spectrum(z_all.reshape(-1, z_all.shape[-1]), body.K)
    derivs = synthesize_derivatives(z_hat_all, body.K, body.N_w, body.L, body.max_order)
    h_nu = body.nu_input_proj(derivs)
    for block in body.nu_blocks:
        h_nu = block(h_nu)
    h_nu = body.nu_final_ln(h_nu)
    nu_field = F.softplus(body.nu_output_proj(h_nu).squeeze(-1))
    print('[$LABEL] fitted nu(x) over real data: min=%.6f  mean=%.6f  max=%.6f  std=%.6f' % (
        nu_field.min().item(), nu_field.mean().item(), nu_field.max().item(), nu_field.std().item()
    ))
"
}

echo "=== [3/6] FULL Jacobian spectrum + conditioning check + fitted beta/nu(x) range on the FINAL propagator ==="
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

echo "=== Section 176 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
