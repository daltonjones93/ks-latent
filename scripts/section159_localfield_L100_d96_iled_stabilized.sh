#!/bin/zsh
# User-directed 2026-09-12. Section 158 (L=100, d_latent=96 via
# n_sites=32/local_channels=3 -- a genuinely local, non-degenerate site
# count, unlike 157's n_sites=4) confirmed the instability is NOT about
# the degenerate site count: pde_head still diverged (non-finite by step
# 6-7, all 10 mixing-check ICs diverged), and TWO things got measurably
# worse: latent D_KY collapsed to ~15.0 (below the L=100 benchmark of
# 21-24), and DA skill dropped to 1.24x (vs. 155's 1.80x, and far below
# this arc's earlier ~10x target). So the failure is inherent to the
# --pde-fix-w-xx-w-xxxx + rich-library mechanism itself, not the site
# count.
#
# Literature detour (2026-09-12): found and read iLED (arXiv:2309.05812,
# "Interpretable Learning of Effective Dynamics for Multiscale Systems"),
# which decomposes learned latent dynamics the SAME way this arc's
# pde_head does -- dz/dt = A_theta*z + Psi_1(z,h) (linear + nonlinear
# closure) -- and reports it does NOT blow up on KS. Their stabilization
# mechanism has two halves:
#   1. A_theta reparametrized W - W^T - diag(|w|), guaranteeing
#      non-positive eigenvalues (a provably stable linear part).
#   2. ||Psi_1(z,h)||^2 penalized directly, keeping the nonlinear closure
#      small relative to the linear part.
# Half 1 was ALREADY covered here, more strongly, by --pde-fix-w-xx-
# w-xxxx (fixes the true KS physical dissipation coefficients exactly,
# not just their sign) -- confirmed by direct analysis: with w_xx=-1,
# w_xxxx=-1 fixed, Re(lambda(k)) = k^2 - k^4 exactly matches true KS's own
# linear spectrum, so the LINEAR part was never the problem. Half 2 had NO
# analogue until now -- nothing in this arc's training loss ever penalized
# the nonlinear closure's own magnitude, and nothing gave it gradient
# pressure against blowing up under its own repeated (autoregressive)
# integration (every prior "pde_distill" term was 1-2 step only).
#
# User: "sure, directly replicate iLED's stabilization mechanism and
# incorporate it into a rollout term to pde_distill loss." Implemented as
# two new mechanisms (ks_latent/models/propagator.py, ks_latent/training/
# loops.py, ks_latent/config.py):
#   - `nonlinear_field()` on _SpectralPDEDeltaBody/_SpectralPDERawDeltaBody:
#     isolates the closure's output restricted to genuinely nonlinear
#     (length>=2) terms, masking the constant/linear columns to zero.
#     `w_pde_nonlinear_l2` penalizes its squared output directly -- the
#     iLED L_non-linearity analogue.
#   - `w_pde_distill_real_rollout`/`pde_distill_real_rollout_k`: pde_head's
#     OWN chained `.rollout()` from a REAL starting state, compared at
#     EVERY step to REAL future states (never aux/propagator's own
#     rollout) -- the iLED L_forecast/L_rec_forecast analogue. Unlike
#     w_pde_rollout ("option B", flagged in Sections 101-106 as "incredibly
#     hard to train" because it chases propagator's own co-adapting
#     rollout), the target here is fixed real data throughout.
# Both verified via direct unit tests (gradient isolation at the masked
# positions: exactly 0.0) and a real smoke-profile training run before
# this launch.
#
# ONLY CHANGE from Section 158 (everything else identical: channel-mean
# fix, Chebyshev basis, --pde-fix-w-xx-w-xxxx, degree=2/max_term_order=7
# 19-term library, masked_mlp_expand attn_window=7, no L1 penalty,
# n_sites=32/local_channels=3, L=100):
#   Stage 1: + --w-pde-nonlinear-l2 0.01 (k_pred_cap=2 by default here, so
#     --pde-distill-real-rollout-k is capped at 2 -- still genuinely
#     multi-step, just short; Stage 2 is where the real rollout horizon
#     lives).
#   Stage 2: + --w-pde-distill-real-rollout 0.3 --pde-distill-real-rollout-k 6
#     (6 steps -- directly targets the exact divergence-onset step 157/158
#     both hit) + --w-pde-nonlinear-l2 0.01.
#
# Verified via real (--profile full, 2-3 epoch) dry runs through BOTH
# stages before this launch -- no errors, pde_distill_real_rollout and
# pde_nonlinear both print and decrease sensibly in the log.
#
# Same before/after Jacobian-spectrum+conditioning check, visualization
# (including the smooth-field GIF), Gate 3/4 suite, k=1,2,4,8
# pde-predictiveness evaluation, standalone pde_head divergence check, the
# Chebyshev-to-monomial coefficient conversion, and the multi-IC
# self-spectrum mixing check as prior sections -- directly comparable
# against 158's own numbers (same L=100/d_latent=96 setup).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points.h5
KS_L=100

TAG=section159_localfield_L100_d96_iled_stabilized_200ep

echo "=== [1/8] Stage 1 (L=100, d_latent=96 via n_sites=32/local_channels=3): local_field + masked_mlp_expand (attn_window=7, expand_factor=3) + --w-channel-mean 0.01 + --pde-distill --pde-mutual --pde-field-kind CHEBYSHEV --pde-poly-degree 2 --pde-max-order 4 --pde-poly-max-term-order 7 --pde-fix-w-xx-w-xxxx + NEW --w-pde-nonlinear-l2 0.01 (iLED L_non-linearity), w_var=0.01/w_spatial=0.06/w_logdet=0.01/w_smooth=0.006/w_pde_distill=0.5, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --dataset "$DATASET" \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone masked_mlp_expand --mode markovian \
  --prop-attn-window 7 --masked-mlp-expand-factor 3 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.06 --spatial-signed --w-logdet 0.01 --w-smooth 0.006 \
  --w-channel-mean 0.01 \
  --pde-distill --w-pde-distill 0.5 --pde-mutual --pde-field-kind chebyshev --pde-poly-degree 2 \
  --pde-max-order 4 --pde-poly-max-term-order 7 --pde-fix-w-xx-w-xxxx \
  --w-pde-nonlinear-l2 0.01 \
  --pde-integrator euler \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/8] Stage 2: propagator AND pde_head continue together, UNFROZEN (mutual), w_pde_distill=0.5 + NEW --w-pde-distill-real-rollout 0.3 --pde-distill-real-rollout-k 6 (iLED L_forecast analogue) + --w-pde-nonlinear-l2 0.01, NO L1 penalty, --amp, k_max=12, 300 epochs (established curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --dataset "$DATASET" \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --init-pdehead-checkpoint "$PDEHEAD_STAGE1" \
  --w-pde-distill 0.5 \
  --w-pde-distill-real-rollout 0.3 --pde-distill-real-rollout-k 6 \
  --w-pde-nonlinear-l2 0.01 \
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

with h5py.File('$DATASET','r') as f:
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

echo "=== [3/8] FULL Jacobian spectrum + conditioning check on the FINAL propagator ==="
_spectrum_check "$STAGE2_PROP" "stage2-final" > artifacts/logs/jacobiancheck_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/jacobiancheck_${STAGE2_TAG}.log

echo "=== [4/8] Visualization + Gate 3/4 (Lyapunov/D_KY, DA skill, D1-D9) -- L=100 ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --dataset "$DATASET" --L $KS_L \
  --rollout-steps 200 --tag "$STAGE2_TAG" \
  > artifacts/logs/visualize_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/visualize_${STAGE2_TAG}.log

mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --L $KS_L \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --L $KS_L \
  --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1

echo "=== [5/8] Is the pde_head actually predictive? Same k=1,2,4,8 evaluation as Sections 138-141/143/145/153/155-158 ==="
mamba run -n da_env python -c "
import h5py, numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$STAGE2_PROP')
pdehead, pdehead_cfg, _ = load_propagator_checkpoint('$STAGE2_PDEHEAD')
ae.eval(); prop.eval(); pdehead.eval()

with h5py.File('$DATASET','r') as f:
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

echo "=== [6/8] Standalone pde_head free-running divergence check ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
pdehead, pdehead_cfg, _ = load_propagator_checkpoint('$STAGE2_PDEHEAD')
ae.eval(); pdehead.eval()
with h5py.File('$DATASET','r') as f:
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
  --dataset "$DATASET" \
  --rollout-steps 55 --tag "$STAGE2_TAG" \
  > artifacts/logs/pde_smooth_field_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/pde_smooth_field_${STAGE2_TAG}.log

echo "=== [7/8] Convert trained Chebyshev-basis poly_coeffs back to ordinary monomial coefficients ==="
mamba run -n da_env python -c "
import torch
from ks_latent.models import load_propagator_checkpoint
from ks_latent.models.propagator import _polynomial_term_indices, _chebyshev_to_monomial_matrix

pdehead, cfg, _ = load_propagator_checkpoint('$STAGE2_PDEHEAD')
inner = pdehead.body.inner
print('field_kind:', inner.field_kind, ' K=%d L=%.1f max_order=%d poly_degree=%d poly_max_term_order=%s' % (
    inner.K, inner.L, inner.max_order, inner.poly_degree, inner.poly_max_term_order
))
char_k = 2 * 3.141592653589793 * inner.K / inner.L
n_vars = inner.max_order + 1
term_indices = _polynomial_term_indices(n_vars, inner.poly_degree, inner.poly_max_term_order)

cheb_weight = inner.poly_coeffs.weight.detach().squeeze(0).clone()
bias = inner.poly_coeffs.bias.detach().item() if inner.poly_coeffs.bias is not None else 0.0
if inner.poly_stable_leading:
    idx = inner._stable_leading_idx
    raw = cheb_weight[idx].item(); sign = inner._stable_leading_sign
    cheb_weight[idx] = sign * (raw ** 2)
if inner.poly_no_constant:
    cheb_weight[0] = 0.0

B = _chebyshev_to_monomial_matrix(n_vars, inner.poly_degree, inner.poly_max_term_order)
mono_weight = B @ cheb_weight

names = ['w', 'w_x', 'w_xx', 'w_xxx', 'w_xxxx', 'w_xxxxx', 'w_xxxxxx']
rows = []
for i, idx in enumerate(term_indices):
    if len(idx) == 0:
        label = '(const)'; denom = 1.0
    else:
        label = '*'.join(names[j] for j in idx)
        denom = 1.0
        for j in idx: denom *= char_k ** j
    rows.append((label, cheb_weight[i].item(), mono_weight[i].item() / denom))
rows.sort(key=lambda r: -abs(r[2]))
print()
print('%-20s %14s %18s' % ('term', 'chebyshev_coeff (raw, TRAINED basis)', 'monomial_coeff (physical, CONVERTED)'))
for label, cheb_c, mono_c in rows[:20]:
    print('%-20s %14.6f %18.6f' % (label, cheb_c, mono_c))
print('... (%d more terms)' % (len(rows) - 20))
print('%-20s %14s %18.6f' % ('bias(intercept)', '', bias))
" > artifacts/logs/chebyshev_monomial_conversion_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/chebyshev_monomial_conversion_${STAGE2_TAG}.log

echo "=== [8/8] pde_head's own self-spectrum mixing check across its standalone rollout, ACROSS MULTIPLE real ICs ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.spectral_field import encode_to_spectrum

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
pdehead, pdehead_cfg, _ = load_propagator_checkpoint('$STAGE2_PDEHEAD')
ae.eval(); pdehead.eval()
K = pdehead_cfg.spectral_K
with h5py.File('$DATASET','r') as f:
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

echo "=== Section 159 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
