#!/bin/zsh
# User-directed 2026-09-12. Section 164 (pde_head fit solely against
# real encoder-derived transitions, MUTUAL so gradient shapes the
# encoder, propagator fully decoupled) was the best combined result of
# the whole arc: never diverges over 200 steps, D_KY=24.27 (dead center
# of the true 21-24 benchmark), dominant coefficient is the true KS
# nonlinearity w*w_x. But its standalone rollout, while never diverging,
# drifted from broadband (t=0) toward 72-86%% concentration in the
# self-FFT DC/mean mode (t=54).
#
# Diagnosed directly (user: "can you recommend a fix to prevent drift
# toward mode-0... please think about it, and implement it"): true KS
# conserves int(u)dx EXACTLY -- every term in -u*u_x-u_xx-u_xxxx is a
# total x-derivative, integrating to zero over a periodic domain. A
# freely-fit polynomial closure has no such guarantee. Measured directly
# on 164's own trained pde_head: field(z).mean(dim=-1) (the predicted
# dz/dt's own spatial mean) across 500 real states was -0.00031+-0.0045
# -- small but systematically nonzero. Two concrete sources in the
# fitted coefficients: the bias/intercept (0.0065, a flat offset with no
# counterpart in true KS) and the w_x*w_x term (coefficient -0.043, the
# LARGEST term that is not a total x-derivative -- w_x^2>=0 pointwise
# always, so its contribution to the mean never oscillates out, unlike
# e.g. w_xx*w_xxx = d/dx(w_xx^2/2), an exact total derivative that
# contributes zero).
#
# Section 165 (still running, killed mid-Stage-1 to launch this instead)
# kept Section 164's --w-spatial/--w-smooth (just doubled per a separate
# request) alongside the new mean-conservation fix. User's reconsidered
# position: "I think the w-spatial term should be removed right? I think
# the pde itself should shape the latent space alone." I.e. spatial_
# coherence_loss/temporal_smoothness_loss are hand-designed priors on
# the latent's own index structure -- with pde_head now MUTUALLY shaping
# the encoder (Section 164 onward), that shaping pressure competes with
# (or substitutes for) whatever these two separate regularizers were
# doing, muddying which mechanism is actually responsible for the
# latent's spatial structure. Consistent with this session's running
# theme (removing --w-pde-distill/--pde-mutual in 163, restricting
# --w-pde-coeff-l1 in 162): let the mechanism whose job this actually is
# do the shaping, rather than stacking a separate hand-designed one on
# top of it.
#
# THIS section removes --w-spatial/--spatial-signed/--w-smooth ENTIRELY
# from Stage 1 (not just reverts 165's doubling -- removes them). Keeps
# EVERYTHING ELSE from Section 165 unchanged: --pde-poly-no-constant,
# --w-pde-mean-conservation 200.0 (both stages), channel-mean fix, no L1,
# pde_head fit solely against real data with Stage-1-only mutual
# encoder-shaping, propagator fully decoupled, iLED rollout+nonlinear-l2
# in Stage 2.
#
# Verified via a real (--profile full, --epochs 3) dry run of Stage 1
# with --w-spatial/--w-smooth removed -- no NaN, trains normally.
#
# Same before/after Jacobian-spectrum+conditioning check, visualization
# (including the smooth-field GIF), Gate 3/4 suite, k=1,2,4,8
# pde-predictiveness evaluation, standalone pde_head divergence check,
# and the multi-IC self-spectrum mixing check (measuring whether the
# mode-0 drift is actually reduced) as Section 164 -- directly
# comparable numbers.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section166_nospatial_nosmooth

echo "=== [1/7] Stage 1: Section 165's recipe MINUS --w-spatial/--spatial-signed/--w-smooth (removed entirely -- pde_head's own mutual shaping is now the ONLY pressure on the latent's spatial structure): local_field + masked_mlp_expand (attn_window=7, expand_factor=3) + --pde-poly-no-constant + --w-pde-mean-conservation 200.0, pde_head fit SOLELY against real encoder-derived transitions, MUTUAL (--w-pde-distill-real 0.5 --pde-distill-real-mutual -- gradient reaches the encoder; NO --w-pde-distill/--pde-mutual -- propagator never seen) (NO --pde-K -- full spectrum, K=d_latent//2+1=49) --pde-poly-max-term-order 6, w_var=0.01/w_logdet=0.01, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone masked_mlp_expand --mode markovian \
  --prop-attn-window 7 --masked-mlp-expand-factor 3 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-logdet 0.01 \
  --w-channel-mean 0.01 \
  --pde-distill --pde-field-kind polynomial --pde-poly-degree 2 \
  --pde-poly-max-term-order 6 --pde-integrator euler --pde-poly-no-constant \
  --w-pde-distill-real 0.5 --pde-distill-real-mutual \
  --w-pde-mean-conservation 200.0 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/7] Stage 2: propagator trains independently (pde_head's loss no longer touches it at all); pde_head continues fitting SOLELY against real data: --w-pde-distill-real 0.5 (single-step) + --w-pde-distill-real-rollout 0.3 --pde-distill-real-rollout-k 4 (multi-step, iLED L_forecast) + --w-pde-nonlinear-l2 0.01 (iLED L_non-linearity) + NEW --w-pde-mean-conservation 200.0, NO --w-pde-distill, grad_clip active, --amp, k_max=12, 300 epochs (established curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --init-pdehead-checkpoint "$PDEHEAD_STAGE1" \
  --w-pde-distill-real 0.5 \
  --w-pde-distill-real-rollout 0.3 --pde-distill-real-rollout-k 4 \
  --w-pde-nonlinear-l2 0.01 \
  --w-pde-mean-conservation 200.0 \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt
STAGE2_PDEHEAD=artifacts/stage2_pdehead_patched_full_${STAGE2_TAG}.pt

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
"
}

echo "=== [3/7] FULL Jacobian spectrum + conditioning check on the FINAL propagator ==="
_spectrum_check "$STAGE2_PROP" "stage2-final" > artifacts/logs/jacobiancheck_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_${STAGE2_TAG}.log

echo "=== [4/7] Visualization + Gate 3/4 (Lyapunov/D_KY, DA skill, D1-D9) ==="
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

echo "=== [5/7] Is the pde_head actually predictive? Same k=1,2,4,8 evaluation as Sections 138-141/143/145/153 ==="
mamba run -n da_env python -c "
import h5py, numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$STAGE2_PROP')
pdehead, pdehead_cfg, _ = load_propagator_checkpoint('$STAGE2_PDEHEAD')
ae.eval(); prop.eval(); pdehead.eval()

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj.shape
with torch.no_grad():
    z_all = ae.encode(traj.reshape(n*T, NX)).reshape(n, T, -1)

k_max = 8
z0 = z_all[:, 0, :]
with torch.no_grad():
    z_prop = prop.rollout(z0, z0, k_max)
    z_pde = pdehead.rollout(z0, z0, k_max)

for k in (1, 2, 4, 8):
    mse_vs_prop = ((z_pde[:, :k] - z_prop[:, :k])**2).mean().item()
    r2_vs_prop = 1 - mse_vs_prop / z_prop[:, :k].var().item()
    print('k=%d: pde_vs_prop_mse=%.5f (R2=%.4f)' % (k, mse_vs_prop, r2_vs_prop))
n_params = sum(p.numel() for p in pdehead.parameters())
print('param count: %d' % n_params)
" > artifacts/logs/jointpde_eval_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jointpde_eval_${STAGE2_TAG}.log

echo "=== [6/7] Standalone pde_head free-running divergence check ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
pdehead, pdehead_cfg, _ = load_propagator_checkpoint('$STAGE2_PDEHEAD')
ae.eval(); pdehead.eval()
with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][0], dtype=torch.float32)
u0 = traj[0].unsqueeze(0)
with torch.no_grad():
    z0 = ae.encode(u0)
    z_roll = pdehead.rollout(z0, z0, 200)
finite = torch.isfinite(z_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('first non-finite step (out of 200, -1 = never):', first_bad)
for t in [0,10,20,30,40,60,80,100,150,199]:
    v = z_roll[0, t]
    print(t, v.abs().max().item() if torch.isfinite(v).all() else 'nan/inf')
" > artifacts/logs/pdehead_standalone_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/pdehead_standalone_${STAGE2_TAG}.log

echo "=== visualize_pde_smooth_field (smooth Hovmoller + GIF) ==="
mamba run -n da_env python scripts/visualize_pde_smooth_field.py \
  --ae-checkpoint "$AE" --pdehead-checkpoint "$STAGE2_PDEHEAD" \
  --rollout-steps 55 --tag "$STAGE2_TAG" \
  > artifacts/logs/pde_smooth_field_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/pde_smooth_field_${STAGE2_TAG}.log

echo "=== [7/7] pde_head's own self-spectrum mixing check across its standalone rollout, ACROSS MULTIPLE real ICs -- directly measures whether the mode-0 drift is reduced vs. Section 164 ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.spectral_field import encode_to_spectrum

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
pdehead, pdehead_cfg, _ = load_propagator_checkpoint('$STAGE2_PDEHEAD')
ae.eval(); pdehead.eval()
K = pdehead_cfg.spectral_K
with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    trajs = torch.tensor(f['trajectories'][:5], dtype=torch.float32)
for run_idx in range(5):
    for start in [0, 100]:
        u0 = trajs[run_idx, start].unsqueeze(0)
        with torch.no_grad():
            z0 = ae.encode(u0)
            z_roll = pdehead.rollout(z0, z0, 55).squeeze(0)
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
" > artifacts/logs/pdehead_mixing_check_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/pdehead_mixing_check_${STAGE2_TAG}.log

echo "=== Section 166 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
