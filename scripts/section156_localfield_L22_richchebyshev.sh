#!/bin/zsh
# User-directed 2026-09-11, following Section 155's result: the
# channel-mean root-cause fix (--w-channel-mean 0.01) clearly improved the
# PROPAGATOR's own dynamics (DA skill 1.30x->1.80x, D_KY/n_positive/lambda1
# all healthy) but broke the SAME pde_head recipe that worked in Section
# 153 (R^2@k=8 collapsed 0.971->0.343, standalone rollout now diverges at
# t=30, all 10 tested initial conditions in the mixing check diverged).
# Diagnosis: 153's pde_head could get away with a good-looking fit partly
# because it was tracking an artificially clean, low-entropy two-mode
# signal (the mode-32 flattening artifact) -- removing that artifact
# exposed genuinely richer dynamics that the same tiny (18-term,
# max_order=4/degree=2/max_term_order=6), L1-linear-only-restricted
# closure can no longer represent. User: "sure run it again and drop
# --pde-coeff-l1-linear-only, allow a richer term library, use chebyshev
# basis too. also use L=22. Let's see if we can fit anything."
#
# FOUR changes from Section 155 (keeping the channel-mean fix):
#
# 1. --pde-coeff-l1-linear-only DROPPED entirely -- along with
#    --w-pde-coeff-l1 itself (both flags removed from both stages). No L1
#    sparsity pressure of any kind on the pde_head's own coefficients --
#    maximum fitting freedom, since the whole point this run is "can we
#    fit ANYTHING", not "can we force a specific sparsity pattern".
#
# 2. Richer library: --pde-poly-degree 2 -> 3 (up to 3-factor products),
#    --pde-poly-max-term-order 6 REMOVED entirely (reverts to the
#    default, unrestricted). --pde-max-order stays at the default 4 (full
#    w..w_xxxx derivative stack, NOT Section 151's restricted max_order=2
#    -- this run wants MORE capacity, not less). Verified directly via
#    _polynomial_term_indices(5, 3, None): 56 terms (up from 18) --
#    substantially richer.
#
# 3. --pde-field-kind chebyshev (Section 152's mechanism) -- verified
#    directly before this launch that the untested degree=3 combination
#    (56-term library) converts correctly: _chebyshev_library vs.
#    _polynomial_library + _chebyshev_to_monomial_matrix agree to
#    relative error ~1e-7 on random inputs.
#
# 4. --dataset switched to the L=22 KS regime (stage1_trajectories_
#    L22_dtsnap1.h5, NX=256 -- same grid resolution as the L=100 dataset,
#    so local_field's own architecture (n_sites=32/local_channels=3/
#    site_mix_radius=2/n_site_mix_layers=3/hidden=32) needs no changes) --
#    a MUCH lower-dimensional, less chaotic regime (true KS benchmark:
#    D_KY~5.2-5.6 at L=22, vs. ~21-24 at L=100) intended to make "can a
#    local closure fit real KS dynamics" more tractable while everything
#    else about the pipeline (encoder architecture, channel-mean fix,
#    propagator backbone) stays the same. ALL diagnostic/analysis
#    scripts below are updated to use the L=22 dataset/domain length
#    consistently (train_stage2's own --dataset, every inline python -c
#    snippet's hardcoded dataset path, run_da_pff.py/run_diagnostics.py/
#    visualize_rollout.py's own --L 22) -- a real, easy-to-miss mismatch
#    risk (per run_da_pff.py's OWN docstring: "MUST match the checkpoint's
#    actual training KSConfig.L or this comparison is meaningless") if
#    left at any script's own default (100.0).
#
# --w-channel-mean 0.01 (Section 155's validated root-cause fix) is KEPT
# unchanged. Otherwise IDENTICAL to Section 155: local_field (n_sites=32/
# local_channels=3/site_mix_radius=2/n_site_mix_layers=3/hidden=32) +
# masked_mlp_expand (attn_window=7, expand_factor=3), --pde-distill
# --pde-mutual --pde-integrator euler (NO --pde-K -- full spectrum),
# w_var=0.01/w_spatial=0.06 signed/w_logdet=0.01/w_smooth=0.006/
# w_pde_distill=0.5, --full-propagator --amp, 200 epochs; Stage 2 unfrozen
# mutual continuation, w_pde_distill=0.5, k_max=12, 300 epochs
# (established curriculum).
#
# Verified via real (--profile full, 2-3 epoch) dry runs through BOTH
# stages on the L=22 dataset before this launch, with the full combined
# recipe (chebyshev + degree=3 + unrestricted term order + no L1 +
# channel-mean, all together) -- no errors.
#
# Same before/after Jacobian-spectrum+conditioning check, visualization
# (including the smooth-field GIF), Gate 3/4 suite, k=1,2,4,8
# pde-predictiveness evaluation, standalone pde_head divergence check, the
# Chebyshev-to-monomial coefficient conversion (Section 152's step), and
# the multi-IC self-spectrum mixing check (Section 155's step) as prior
# sections -- directly comparable in KIND, though NOT in absolute D_KY/
# lambda1/etc. magnitude against L=100 runs, since L=22's own true
# benchmark values are intrinsically smaller.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
KS_L=22

TAG=section156_localfield_L22_richchebyshev_200ep

echo "=== [1/8] Stage 1 (L=22): local_field + masked_mlp_expand (attn_window=7, expand_factor=3) + --w-channel-mean 0.01 (kept) + --pde-distill --pde-mutual --pde-field-kind CHEBYSHEV --pde-poly-degree 3 (56-term richer library, NO --pde-poly-max-term-order restriction, NO L1 penalty at all), w_var=0.01/w_spatial=0.06/w_logdet=0.01/w_smooth=0.006/w_pde_distill=0.5, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --dataset "$DATASET" \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone masked_mlp_expand --mode markovian \
  --prop-attn-window 7 --masked-mlp-expand-factor 3 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.06 --spatial-signed --w-logdet 0.01 --w-smooth 0.006 \
  --w-channel-mean 0.01 \
  --pde-distill --w-pde-distill 0.5 --pde-mutual --pde-field-kind chebyshev --pde-poly-degree 3 \
  --pde-integrator euler \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/8] Stage 2 (L=22): propagator AND pde_head continue together, UNFROZEN (mutual), w_pde_distill=0.5, NO L1 penalty, --amp, k_max=12, 300 epochs (established curriculum) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --dataset "$DATASET" \
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

echo "=== [4/8] Visualization + Gate 3/4 (Lyapunov/D_KY, DA skill, D1-D9) -- L=22 ==="
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

echo "=== [5/8] Is the pde_head actually predictive? Same k=1,2,4,8 evaluation as Sections 138-141/143/145 ==="
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

echo "=== Section 156 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
