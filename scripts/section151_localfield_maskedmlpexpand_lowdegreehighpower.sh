#!/bin/zsh
# User-directed 2026-09-11, following the direct measurement (Section 150's
# own investigation) that the z-lowpass regularizer approach (149/150)
# reshaped z's self-spectrum into a middle-wavenumber "hump" instead of a
# genuine low-pass decay, which structurally hurt the polynomial closure's
# fit (R^2@k=8 collapsed from 0.97+ to ~0.47-0.50 in both 149/150) -- a
# closure built from LOW+HIGH order derivatives (w, w_xxxx) is naturally
# suited to a BIMODAL spectrum (148's own baseline: 78% energy in modes
# 0-4, 21% in modes 25-48, ~nothing between), not a mid-band-dominated
# one. User: "well then I want to try something else. why don't we limit
# the pde to have at most up to two derivatives, but keep higher powers of
# products of those derivatives. so that a term like w^2 * w_xx would be
# allowed or w * w_x^2 w_xx etc. let's limit the polynomial to have at
# most 3 values though but total degree 5, otherwise the polynomial basis
# grows too large."
#
# Returns to Section 148's own proven-healthy baseline (local_field +
# masked_mlp_expand attn_window=7, NO z-lowpass, NO --pde-K truncation --
# D_KY=23.08/n_positive=12/lambda1=0.089, D3 significant, skill=1.31x,
# pde R^2@k=8=0.975) and changes ONLY the pde_head's own polynomial
# structure, isolating this one variable cleanly:
#
#   OLD (148, and every --pde-field-kind polynomial run before this):
#     --pde-max-order 4 (derivatives up to w_xxxx) --pde-poly-degree 2
#     (up to 2-factor products) --pde-poly-max-term-order 6 (combined
#     order <=5) -> terms like w_xx, w_xxxx, w_xxxx*w_xxxx (excluded at
#     this threshold), w*w_x, etc.
#
#   NEW (151): --pde-max-order 2 (ONLY w/w_x/w_xx -- w_xxx/w_xxxx dropped
#     entirely from the library, not just de-weighted) --pde-poly-degree 3
#     (up to 3-factor products, e.g. w*w_x*w_xx, w_x^3) --pde-poly-
#     max-term-order 6 (combined order <=5, matching the user's "total
#     degree 5" -- verified directly: with n_vars=3/degree=3/
#     max_term_order=6 this keeps exactly 19 terms, e.g. w^2*w_xx
#     (sum=2), w*w_x^2*w_xx is NOT reachable at degree<=3 with only 2
#     distinct low-order factors allowed per term but w_x*w_xx^2 (sum=5)
#     IS kept -- the single EXCLUDED degree-3 term is w_xx^3 (sum=6).
#     _polynomial_term_indices/_polynomial_library already supported
#     degree=3 (added earlier this session, 2026-09-09) -- the ONLY code
#     change needed was widening --pde-poly-degree's CLI choices from
#     [1,2] to [1,2,3] (scripts/train_stage1_patched.py), the config-level
#     validation already accepted 1/2/3 natively.
#
# Tests directly whether ALLOWING higher powers/products of just a FEW
# low-order derivatives (a "richer nonlinearity in fewer variables"
# closure) fits better than the previous "more derivative orders, capped
# at pairwise products" design -- a genuinely different point in the
# same design space, motivated by 150's own finding that middle-order
# derivative content (w_x, w_xx) carries real signal this arc's spectra
# often have, which the OLD design under-served (only pairwise products
# of up to 5 variables, so e.g. w_x^3 or w*w_x*w_xx were never
# representable at all).
#
# Otherwise IDENTICAL to Section 148: local_field (n_sites=32/
# local_channels=3/site_mix_radius=2/n_site_mix_layers=3/hidden=32) +
# masked_mlp_expand (attn_window=7, expand_factor=3), --pde-distill
# --pde-mutual --pde-integrator euler, w_var=0.01/w_spatial=0.06 signed/
# w_logdet=0.01/w_smooth=0.006/w_pde_distill=0.5, --full-propagator --amp,
# 200 epochs; Stage 2 unfrozen mutual continuation, w_pde_distill=0.5,
# k_max=12, 300 epochs (established curriculum).
#
# Verified via a real (--profile full, 2-epoch) dry run through BOTH
# stages before this launch -- --pde-max-order 2/--pde-poly-degree 3
# accepted (19-term library confirmed via a direct _polynomial_term_indices
# check), pde_distill loss computes and decreases normally, no errors.
#
# Same before/after Jacobian-spectrum+conditioning check, visualization
# (including the smooth-field GIF), Gate 3/4 suite, k=1,2,4,8
# pde-predictiveness evaluation, and standalone pde_head divergence check
# as Sections 141/143/145/148 -- directly comparable numbers.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section151_localfield_maskedmlpexpand_lowdegreehighpower_200ep

echo "=== [1/6] Stage 1: local_field + masked_mlp_expand (attn_window=7, expand_factor=3) + --pde-distill --pde-mutual NEW pde_head structure (--pde-max-order 2, only w/w_x/w_xx; --pde-poly-degree 3, up to 3-factor products; --pde-poly-max-term-order 6, combined order <=5 -> 19 terms), w_var=0.01/w_spatial=0.06/w_logdet=0.01/w_smooth=0.006/w_pde_distill=0.5, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone masked_mlp_expand --mode markovian \
  --prop-attn-window 7 --masked-mlp-expand-factor 3 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.06 --spatial-signed --w-logdet 0.01 --w-smooth 0.006 \
  --pde-distill --w-pde-distill 0.5 --pde-mutual --pde-field-kind polynomial \
  --pde-max-order 2 --pde-poly-degree 3 --pde-poly-max-term-order 6 --pde-integrator euler \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/6] Stage 2: propagator AND pde_head continue together, UNFROZEN (mutual), w_pde_distill=0.5, --amp, k_max=12, 300 epochs (established curriculum) ==="
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

echo "=== [3/6] FULL Jacobian spectrum + conditioning check on the FINAL propagator ==="
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

echo "=== [5/6] Is the pde_head actually predictive? Same k=1,2,4,8 evaluation as Sections 138-141/143/145 ==="
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

echo "=== [6/6] Standalone pde_head free-running divergence check ==="
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

echo "=== Section 151 complete ==="
tail -3 artifacts/logs/stage2_${STAGE2_TAG}.log
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${STAGE2_TAG}.log || true
grep -E "skill_free_over_da|calibration" artifacts/logs/gate3_da_${STAGE2_TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log || true
