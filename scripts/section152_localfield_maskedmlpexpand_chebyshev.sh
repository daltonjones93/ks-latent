#!/bin/zsh
# User-directed 2026-09-11, following a discussion of whether a Chebyshev
# basis could represent the pde_head's closure advantageously: "please try
# the chebyshev to monomial basis conversion you mentioned next in 152.
# train in chebyshev space, then transform back. you can use the setup
# from 148."
#
# Section 148 (local_field + masked_mlp_expand attn_window=7, NO z-lowpass,
# NO --pde-K truncation) is the last confirmed-healthy baseline this arc:
# D_KY=23.08/n_positive=12/lambda1=0.089, D3 significant, skill=1.31x,
# pde R^2@k=8=0.975 with field_kind="polynomial" (ordinary monomials).
#
# ONE change from Section 148: --pde-field-kind polynomial -> chebyshev
# (brand-new mechanism built this session: ks_latent.models.propagator.
# _chebyshev_library, alongside _polynomial_library -- SAME term
# structure/count/max_term_order filtering, `_polynomial_term_indices` is
# reused UNCHANGED, only each term's factors are evaluated as Chebyshev
# polynomials T_k(d_i) -- via the standard three-term recurrence, valid
# for any real input, see that function's docstring for the caveat that
# the classical conditioning benefit is guaranteed only on [-1,1] -- of
# the repeated derivative channel instead of plain powers d_i^k; terms
# with no repeated variable, i.e. multiplicity 1, are UNCHANGED since
# T_1(x)=x). --pde-poly-degree/--pde-poly-max-term-order/--pde-max-order
# all unchanged from 148 (2/6/4 respectively) -- ONLY the basis functions
# differ, term-for-term identical structure.
#
# AFTER training, converts the learned Chebyshev-basis poly_coeffs.weight
# back to ordinary, directly-interpretable monomial coefficients via
# ks_latent.models.propagator._chebyshev_to_monomial_matrix -- a genuine,
# EXACT change of basis (verified numerically to machine precision,
# max relative error ~8e-8, before this launch: sampling random z,
# confirming _chebyshev_library(z) @ chebyshev_coeffs ==
# _polynomial_library(z) @ (B @ chebyshev_coeffs) for random
# chebyshev_coeffs), not an approximation -- so the reported monomial
# coefficients represent EXACTLY the function that was actually trained,
# directly comparable to every prior extraction's own monomial table
# (138/140/141/143/145/148/151) and to true KS's own -w*w_x-w_xx-w_xxxx.
#
# Tests whether training IN the (potentially better-conditioned)
# Chebyshev basis produces a better-fitting closure than training
# directly in the monomial basis (148's own R^2@k=8=0.975 is already
# quite high, so the headroom to measure here is modest but real) --
# and, separately, whether the CONVERTED-BACK monomial coefficients still
# show the same qualitative pattern (w_xx/w_xxxx dominant, correctly
# signed, w*w_x small-but-correct) this arc's other extractions have
# found.
#
# Otherwise IDENTICAL to Section 148: local_field (n_sites=32/
# local_channels=3/site_mix_radius=2/n_site_mix_layers=3/hidden=32) +
# masked_mlp_expand (attn_window=7, expand_factor=3), --pde-distill
# --pde-mutual --pde-poly-max-term-order 6 --pde-integrator euler (NO
# --pde-K -- full spectrum), w_var=0.01/w_spatial=0.06 signed/
# w_logdet=0.01/w_smooth=0.006/w_pde_distill=0.5, --full-propagator --amp,
# 200 epochs; Stage 2 unfrozen mutual continuation, w_pde_distill=0.5,
# k_max=12, 300 epochs (established curriculum).
#
# Verified via a real (--profile full, 2-epoch) dry run through BOTH
# stages before this launch -- --pde-field-kind chebyshev accepted,
# pde_distill loss computes and decreases normally, no errors -- AND via
# the standalone numerical basis-conversion check described above.
#
# Same before/after Jacobian-spectrum+conditioning check, visualization
# (including the smooth-field GIF), Gate 3/4 suite, k=1,2,4,8
# pde-predictiveness evaluation, and standalone pde_head divergence check
# as Sections 141/143/145/148/151 -- directly comparable numbers -- PLUS
# a new [7/7] step converting and printing the monomial-equivalent
# coefficient table.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section152_localfield_maskedmlpexpand_chebyshev_200ep

echo "=== [1/7] Stage 1: local_field + masked_mlp_expand (attn_window=7, expand_factor=3) + --pde-distill --pde-mutual --pde-field-kind CHEBYSHEV (NEW -- same term structure as 148's polynomial, Chebyshev basis functions) (NO --pde-K -- full spectrum, K=d_latent//2+1=49) --pde-poly-max-term-order 6, w_var=0.01/w_spatial=0.06/w_logdet=0.01/w_smooth=0.006/w_pde_distill=0.5, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone masked_mlp_expand --mode markovian \
  --prop-attn-window 7 --masked-mlp-expand-factor 3 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.06 --spatial-signed --w-logdet 0.01 --w-smooth 0.006 \
  --pde-distill --w-pde-distill 0.5 --pde-mutual --pde-field-kind chebyshev --pde-poly-degree 2 \
  --pde-poly-max-term-order 6 --pde-integrator euler \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/7] Stage 2: propagator AND pde_head continue together, UNFROZEN (mutual), w_pde_distill=0.5, --amp, k_max=12, 300 epochs (established curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --init-pdehead-checkpoint "$PDEHEAD_STAGE1" \
  --w-pde-distill 0.5 \
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

echo "=== [5/7] Is the pde_head actually predictive? Same k=1,2,4,8 evaluation as Sections 138-141/143/145 ==="
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

echo "=== [7/7] Convert trained Chebyshev-basis poly_coeffs back to ordinary monomial coefficients ==="
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
bias = inner.poly_coeffs.bias.detach().item()
if inner.poly_stable_leading:
    idx = inner._stable_leading_idx
    raw = cheb_weight[idx].item(); sign = inner._stable_leading_sign
    cheb_weight[idx] = sign * (raw ** 2)

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
print('%-15s %14s %18s' % ('term', 'chebyshev_coeff (raw, TRAINED basis)', 'monomial_coeff (physical, CONVERTED)'))
for label, cheb_c, mono_c in rows:
    print('%-15s %14.6f %18.6f' % (label, cheb_c, mono_c))
print('%-15s %14s %18.6f' % ('bias(intercept)', '', bias))
" > artifacts/logs/chebyshev_monomial_conversion_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/chebyshev_monomial_conversion_${STAGE2_TAG}.log

echo "=== Section 152 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
