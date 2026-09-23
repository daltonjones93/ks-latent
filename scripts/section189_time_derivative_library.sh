#!/bin/zsh
# User-directed 2026-09-18: "can we incorporate time derivatives into the
# pde polynomial? might give us a richer expression. We can approximate
# them using rollout terms potentially."
#
# Required real implementation work, not just a config flag (see
# ks_latent/models/propagator.py, ks_latent/config.py,
# scripts/train_stage1_patched.py -- Section 189):
#   1. NEW `AuxPropagatorConfig.spectral_poly_time_deriv` (+ `_dt_snap`):
#      appends a finite-difference feature `w_t_fd = (w - w_prev) /
#      dt_snap` to the polynomial/Chebyshev library (`_SpectralPDEDeltaBody.
#      _augment_derivs_with_time_deriv`), so terms like `w_t`, `w*w_t`,
#      `w_t*w_xxxx`, `w_t^2` can appear in the fitted PDE.
#   2. `w_prev` is a genuine previous-step STATE, not a second implicit
#      unknown -- exactly the user's own "approximate them using rollout
#      terms" framing: `LatentPropagator.step()` was updated to stop
#      discarding `z_prev` for markovian mode when this flag is active
#      (previously ALWAYS ignored for markovian -- a real, pre-existing
#      gap). `.rollout()` already threads consecutive `(z_prev, z_curr)`
#      pairs through its own loop, so this "just works" for the K-step
#      curriculum training loss with NO changes to loops.py: the first
#      predicted step of each window sees a zero fallback (no real
#      z_{n-2} available at markovian window construction), every step
#      after that sees the model's OWN prior prediction as `z_prev` --
#      genuinely using rollout terms, not real finite-differenced ground
#      truth beyond the first step, matching the user's own suggestion.
#   3. `step_one(z)` (single-argument, used by EVERY Jacobian-based
#      regularizer from 183-186: delta_cap, kernel_unstable_floor, both
#      spectrum-shape losses) is UNCHANGED -- no `z_prev` to give, so the
#      new feature falls back to an architectural zero there. Verified
#      directly: gradients flow correctly through the new `z_prev` path
#      (nonzero once poly_coeffs stop being zero-init) and the rollout's
#      own autoregressive `z_prev` threading works end-to-end.
#   4. Fixed a REAL bug found while building this (not present in any
#      prior section): `_polynomial_term_indices`'s `max_term_order`
#      budget sums raw variable INDICES as a proxy for spatial derivative
#      order (correct for w..w_xxxx, where index IS the order) -- but the
#      new w_t variable's index (`max_order+1`, e.g. 5) has no such
#      meaning, so at max_term_order=5 EVERY term containing w_t was
#      silently excluded (w_t alone already "cost" 5). Fixed via a new
#      `free_index` parameter (`_polynomial_term_indices`/`_polynomial_
#      library`/`_chebyshev_library`/`_chebyshev_to_monomial_matrix`):
#      the time-derivative variable's contribution to the order-sum is
#      treated as 0 (it is not a spatial derivative, so has no natural
#      order in that budget), while still counting toward the ordinary
#      `poly_degree` factor-count cap. Verified directly: term count with
#      degree=2/max_term_order=5 goes from 15 (without) to 22 (with --
#      the 7 new terms are w_t, w*w_t, w_x*w_t, w_xx*w_t, w_xxx*w_t,
#      w_xxxx*w_t, w_t^2, exactly the expected set).
#
# Recipe: IDENTICAL to Section 187 (Chebyshev, degree=2/max_term_order=5,
# soft-stable w_xx/w_xxxx, exclude-nonconservative, no-constant, delta_cap
# 0.5, two-sided + multistep k=10 spectrum-shape) at L=100, PLUS
# --spectral-poly-time-deriv. This isolates the ONE new variable as the
# sole change from 187's own already-run baseline, so any difference in
# outcome is attributable to it specifically.
#
# Verified via a real dry run (3 epochs, full training-loop path through
# train_stage1_patched.py, not just direct model construction) before this
# launch: no nan, term count 22 as expected, gradients flow through the
# new z_prev path.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section189_time_derivative_library

echo "=== [1/2] Stage 1 ONLY: identical to 187 + NEW --spectral-poly-time-deriv (adds w_t_fd = (w-w_prev)/dt_snap to the Chebyshev library, 22 terms instead of 15) ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 \
  --spectral-field-kind chebyshev --spectral-poly-degree 2 --spectral-poly-max-term-order 5 \
  --spectral-poly-stable-w-xx-w-xxxx --stable-linear-lr-factor 0.02 \
  --spectral-poly-exclude-nonconservative --spectral-poly-no-constant \
  --spectral-poly-time-deriv --spectral-poly-time-deriv-dt-snap 1.0 \
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

echo "=== [2/2] Diagnostics: fitted PDE coefficients (including w_t terms), standalone + multi-IC rollout, Jacobian (one-step and k=10-step composed) ==="
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

print('FINAL Stage-1 checkpoint (epoch 199/200, field_kind=%s, poly_time_deriv=%s, term count=%d):' % (
    body.field_kind, body.poly_time_deriv, body.poly_coeffs.weight.shape[1]))
n_vars = body.max_order + 1 + (1 if body.poly_time_deriv else 0)
free_indices = body._time_deriv_free_indices
term_indices = _polynomial_term_indices(n_vars, body.poly_degree, body.poly_max_term_order, free_indices)
cheb_weight = body._poly_weight().detach().squeeze(0).clone()
char_k = 2 * math.pi * body.K / body.L
B = _chebyshev_to_monomial_matrix(n_vars, body.poly_degree, body.poly_max_term_order, free_indices)
mono_weight = B @ cheb_weight
names = ['w', 'w_x', 'w_xx', 'w_xxx', 'w_xxxx', 'w_xxxxx', 'w_xxxxxx']
if body.poly_time_deriv:
    names = names[:body.max_order + 1] + ['w_t']
rows = []
for i, idx in enumerate(term_indices):
    if len(idx) == 0:
        label = '(const)'; denom = 1.0
    else:
        label = '*'.join(names[j] for j in idx)
        # w_t has no char_k scaling (it isn't a spatial derivative) -- denom
        # contribution from it is 1.0, matching _deriv_norm_scales' own choice.
        denom = 1.0
        for j in idx:
            if free_indices is not None and j in free_indices:
                continue
            denom *= char_k ** j
    rows.append((label, cheb_weight[i].item(), mono_weight[i].item() / denom))
rows.sort(key=lambda r: -abs(r[2]))
print('term'.ljust(20), 'chebyshev_coeff'.rjust(16), 'monomial_coeff(physical)'.rjust(26))
for label, c, m in rows:
    print(label.ljust(20), ('%.6f' % c).rjust(16), ('%.6f' % m).rjust(26))
print()
print('w_t-involving terms specifically:')
for label, c, m in rows:
    if 'w_t' in label:
        print(' ', label.ljust(20), ('%.6f' % c).rjust(16), ('%.6f' % m).rjust(26))

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj20 = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj20.shape
with torch.no_grad():
    z_all = ae.encode(traj20.reshape(n*T, NX)).reshape(n, T, -1)
z0b = z_all[:,0,:]

with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=150)
finite = torch.isfinite(traj_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print()
print('multi-IC (20) rollout first non-finite step:', first_bad)
maxz_per_t = traj_roll.abs().amax(dim=(0,2))
for t in [0,5,10,20,40,60,90,120,149]:
    print(' t=%d  max|z| across all 20 ICs: %.4f   per-IC max|z| range: [%.4f, %.4f]' % (
        t, maxz_per_t[t].item(),
        traj_roll[:,t,:].abs().amax(dim=-1).min().item(),
        traj_roll[:,t,:].abs().amax(dim=-1).max().item(),
    ))
final_states = traj_roll[:,-1,:]
pairwise = torch.cdist(final_states, final_states)
print('final-state pairwise distance (min excl. diag, max, mean):',
      (pairwise + torch.eye(20)*1e9).min().item(), pairwise.max().item(), pairwise.mean().item())
deltas = (traj_roll[:,1:,:] - traj_roll[:,:-1,:]).norm(dim=-1)
print('per-step delta norm at t=5,50,100,148:', deltas[:,5].mean().item(), deltas[:,50].mean().item(), deltas[:,100].mean().item(), deltas[:,148].mean().item())

z0j = z_all[0,0]
J1 = jacrev(lambda z: prop.step_one(z.unsqueeze(0)).squeeze(0))(z0j)
sv1 = torch.linalg.svdvals(J1)
print()
print('ONE-STEP: singular values >= 1.0:', int((sv1>=1.0).sum()), 'out of', sv1.numel())
print('ONE-STEP top 15:', [round(x,3) for x in sv1[:15].tolist()])

def step10(z):
    out = z
    for _ in range(10):
        out = prop.step_one(out.unsqueeze(0)).squeeze(0)
    return out
J10 = jacrev(step10)(z0j)
sv10 = torch.linalg.svdvals(J10)
print('K=10-STEP COMPOSED: top 15 singular values:', [round(x,3) for x in sv10[:15].tolist()])
print('K=10-STEP COMPOSED: targets were 1.1^10=%.3f (top13) / 0.6^10=%.6f (rest)' % (1.1**10, 0.6**10))
" > artifacts/logs/diagnostics_${TAG}.log 2>&1
cat artifacts/logs/diagnostics_${TAG}.log

echo "=== [bonus] Rollout visualization ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$AUX" \
  --rollout-steps 200 --tag "${TAG}_phase1" \
  > artifacts/logs/visualize_${TAG}.log 2>&1
cat artifacts/logs/visualize_${TAG}.log

echo "=== [3/3] Real Lyapunov spectrum (Benettin/QR, full d_latent=96 directions) -- decisive chaos-vs-not check, same as Section 187's ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$AUX')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][3], dtype=torch.float32)
with torch.no_grad():
    z01 = ae.encode(traj[:2]).numpy()

res = lyapunov_spectrum_latent_propagator(
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=400, qr_every=5,
    dt_snap=1.0, warmup_steps=50, seed=0, max_abs_state=1e3,
)
print('lambda1 (physical time units):', res.exponents[0])
print('n_positive:', res.n_positive, '/', res.n_directions)
print('spectrum (first 20):', [round(x,4) for x in res.exponents[:20]])
print('spectrum (last 10):', [round(x,4) for x in res.exponents[-10:]])
try:
    print('D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('D_KY: could not bracket --', e)
" > artifacts/logs/lyapunov_${TAG}.log 2>&1
cat artifacts/logs/lyapunov_${TAG}.log

echo "=== Section 189 complete ==="
