#!/bin/zsh
# User-directed 2026-09-18, following Section 186's result. 186 combined
# BOTH new regularizers from 185's diagnosis (two-sided spectrum-shape +
# a NEW multistep composed-Jacobian spectrum-shape, directly targeting
# "expansive singular vectors feeding into other expansive singular
# vectors") on top of 183's forced_burgers recipe -- and it landed on the
# IDENTICAL failure signature as every one of 180-185: linear growth to
# the delta_cap ceiling (max|z| 3.0->93.2 over 200 steps). The k=10-step
# composed Jacobian's own top singular values (measured directly:
# [80.5, 55.0, 46.0, 22.0, ...] against a target of 1.1^10=2.59) confirm
# real persistent self-feeding is happening -- some mixing occurs (naive
# s^10 compounding of the measured one-step top value 4.4 would give
# ~2.7e6, not 80.5), but nowhere near enough to reach bounded behavior.
#
# SEVEN consecutive variants (180-186) spanning cap shape, physical-term
# learnability, hyperviscosity, forcing removal, an unstable-mode floor,
# and now two independent spectrum-shape regularizers, ALL failed
# identically within the forced_burgers family -- which has only ~5
# scalar physical parameters (A, beta, width, nu, mu) controlling the
# entire closure. User: "I think it might be worth trying this
# combination of regularizers with a pde with more degrees of freedom.
# Maybe the polynomial fit pde that we were using before (with the
# chebyshev basis.)"
#
# This is a genuinely different family, from an EARLIER arc (Sections
# 152-160, `_SpectralPDEDeltaBody`'s field_kind in ("polynomial",
# "chebyshev")): a dense SINDy-style linear regression over a library of
# Chebyshev-polynomial products of the derivative stack (w, w_x, w_xx,
# w_xxx, w_xxxx), degree<=2, term budget controlled by
# --spectral-poly-max-term-order. That arc found this closure family
# SEVERELY prone to blowup as a live autoregressive map when combined
# with hard-fixed true linear coefficients + a rich (17-19 term) library
# (Sections 155/157/158/159 -- ALL diverged non-finite within 6-7 steps,
# even with iLED-style stabilization added on top). BUT: that whole arc
# used a DIFFERENT architecture (pde_head trained as a DISTILLATION
# target of a separate masked_mlp_expand live propagator -- pde_head's
# own rollout was never actually driving the training loss) and a
# DIFFERENT regularizer toolkit (L1 coefficient sparsity, iLED nonlinear-
# magnitude penalty, real-rollout distillation) -- it never had
# delta_cap (a hard per-step bound, from the forced_burgers lineage) or
# either of this window's two spectrum-shape mechanisms. So this is a
# genuinely new combination, not a retry of something already known to
# fail under the SAME toolkit.
#
# THIS section: field_kind="chebyshev" as the LIVE propagator for the
# first time (previously only exercised via --pde-field-kind for the
# pde_head distillation target since Section 152) -- required widening
# --spectral-field-kind's CLI choices (ks_latent/models/propagator.py's
# underlying _SpectralPDEDeltaBody/config validator already supported it
# end-to-end; only scripts/train_stage1_patched.py's argparse `choices`
# list was missing it).
#
# Design choices, informed directly by the 152-160 instability history:
#   - degree=2, max_term_order=5 -> 15 terms (verified via
#     _polynomial_term_indices(5,2,5)) -- meaningfully richer than
#     forced_burgers' 5 scalars, but well below the 17-19-term libraries
#     that diverged in 156/158/159. Conservative first attempt.
#   - --spectral-poly-stable-w-xx-w-xxxx (the SOFT mechanism, Section
#     174): w_xx/w_xxxx initialized at the true KS values (-1,-1),
#     architecturally sign-guaranteed negative, but LEARNABLE magnitude
#     -- matches this whole arc's established preference (183: "let A
#     and beta be trainable") over hard-freezing, and gives the correct
#     linear instability spectrum at init rather than hoping training
#     finds it (the actual root cause diagnosed back in 169-173).
#     Paired with --stable-linear-lr-factor 0.02 (default, small steps).
#   - --spectral-poly-exclude-nonconservative + --spectral-poly-no-constant:
#     both EXACT architectural fixes matching true KS's own conservation
#     law and lack of a constant term -- free correctness, zero cost.
#   - --prop-delta-cap 0.5: kept as the safety net (this family never had
#     it in 152-160; it may be exactly what that arc was missing to
#     survive rather than diverging to nan outright).
#   - Both Section 186 regularizers kept, unchanged: --w-spectrum-shape
#     0.4 --spectrum-shape-two-sided (n_expand=13/target=1.1/floor=0.6)
#     and --w-spectrum-shape-multistep 0.4 (k=10) -- these are
#     step_fn-agnostic (only need aux.step_one), so they apply completely
#     unchanged to this new field_kind.
#   - --w-kernel-unstable-floor NOT set (stays 0.0, off) -- that
#     mechanism (kernel_Lhat()) is forced_burgers-only and RAISES if
#     called on a non-forced_burgers body; leaving the weight at 0.0
#     means the training loop never calls kernel_Lhat() at all (guarded
#     by `if cfg.w_kernel_unstable_floor > 0`), so this is safe by
#     construction, not by omission -- verified by reading the guard in
#     ks_latent/training/loops.py before writing this script.
#   - Everything else identical to 183/185/186: local_field encoder
#     (n_sites=32/channels=3), k_pred_max=10, spatial/logdet/smooth/
#     channel-mean regularizers, --full-propagator --amp, 200 epochs.
#
# Verified via a real dry run (8 epochs, tag section187_dryrun) before
# this launch: no nan, cost comparable to 186 (Chebyshev library adds a
# few more matmul columns vs. burgers' hand-coded terms, but 15 terms is
# small). Per this arc's established caveat (184's own dry run looked
# healthy before diverging at 200 epochs), this is a smoke test only --
# the real 200-epoch run is the actual test of whether more degrees of
# freedom can escape the persistent-self-feeding failure the last seven
# sections all hit.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section187_chebyshev_richer_pde

echo "=== [1/2] Stage 1 ONLY: local_field encoder + spectral_pde_raw propagator, field_kind=chebyshev (degree=2, max_term_order=5, 15 terms) + soft-stable w_xx/w_xxxx + exclude-nonconservative + no-constant + delta_cap 0.5 + two-sided/multistep spectrum-shape (from 186) + k_pred_max=10, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 \
  --spectral-field-kind chebyshev --spectral-poly-degree 2 --spectral-poly-max-term-order 5 \
  --spectral-poly-stable-w-xx-w-xxxx --stable-linear-lr-factor 0.02 \
  --spectral-poly-exclude-nonconservative --spectral-poly-no-constant \
  --spectral-integrator euler \
  --prop-delta-cap 0.5 \
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

echo "=== [2/2] Diagnostics on the Stage-1-only checkpoint: fitted PDE coefficients (Chebyshev->monomial), standalone rollout, Jacobian (one-step AND k=10-step composed), multi-IC separation ==="
mamba run -n da_env python -c "
import math
import h5py, torch
from torch.func import jacrev
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.propagator import _polynomial_term_indices, _chebyshev_to_monomial_matrix

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$AUX')
ae.eval(); prop.eval()
body = prop.body.inner if hasattr(prop.body, 'inner') else prop.body

print('FINAL Stage-1 checkpoint (epoch 199/200, field_kind=%s, degree=%d, max_term_order=%s):' % (
    body.field_kind, body.poly_degree, body.poly_max_term_order))
n_vars = body.max_order + 1
term_indices = _polynomial_term_indices(n_vars, body.poly_degree, body.poly_max_term_order)
cheb_weight = body._poly_weight().detach().squeeze(0).clone()
char_k = 2 * math.pi * body.K / body.L
B = _chebyshev_to_monomial_matrix(n_vars, body.poly_degree, body.poly_max_term_order)
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
print('term'.ljust(20), 'chebyshev_coeff'.rjust(16), 'monomial_coeff(physical)'.rjust(26))
for label, c, m in rows:
    print(label.ljust(20), ('%.6f' % c).rjust(16), ('%.6f' % m).rjust(26))

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][0], dtype=torch.float32)
u0 = traj[0].unsqueeze(0)
with torch.no_grad():
    z0 = ae.encode(u0)
    z_roll = prop.rollout(z0, z0, 200)
finite = torch.isfinite(z_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print()
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
print()
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

echo "=== Section 187 Stage 1 complete ==="
