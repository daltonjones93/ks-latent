#!/bin/zsh
# User-directed 2026-09-18: "that might be the next thing to try,
# incorporate u_tt" -- following up on Section 189's w_t library
# extension (finite-difference first time derivative).
#
# Required real implementation work (ks_latent/models/propagator.py,
# ks_latent/config.py, scripts/train_stage1_patched.py -- Section 190):
#   1. NEW `AuxPropagatorConfig.spectral_poly_time_deriv2` (independent of
#      `spectral_poly_time_deriv` -- either may be on without the other):
#      appends a SECOND library variable, `w_tt_fd = (w - 2*w_prev +
#      w_prev2) / dt_snap**2` -- the standard 3-point BACKWARD second
#      difference, using only PAST states (no future state needed),
#      matching the same "approximate them using rollout terms" framing
#      as w_t itself.
#   2. Needs a genuine THIRD state (`z_prev2`, two steps back) in addition
#      to `z_prev` -- generalized `LatentPropagator.step()`/`.rollout()`
#      to thread an optional `z_prev2` kwarg alongside `z_prev`.
#      `.rollout()` now maintains a 3-wide sliding window internally
#      (`z_prev2, z_prev, z_curr = z_prev, z_curr, z_next` each
#      iteration, generalizing the EXISTING 2-wide pattern `w_t` already
#      used) -- so, exactly like `w_t`, this "just works" for the K-step
#      curriculum training loss with NO changes to loops.py: only the
#      very first predicted step sees the architectural zero fallback for
#      w_tt (neither z_prev nor z_prev2 reflects real history yet); the
#      SECOND predicted step onward gets a genuine (if `z_prev`==`z_prev2`
#      at that point, since markovian windowing duplicates the initial
#      pair) two-step-back state, and every step after that is fully
#      real, self-generated rollout history.
#   3. Generalized `_polynomial_term_indices`'s `free_index: int|None`
#      (Section 189) to `free_indices: frozenset[int]|None` (Section 190)
#      so BOTH `w_t` and `w_tt` can be exempted from the `max_term_order`
#      combined-order budget simultaneously (same reasoning as 189: they
#      are not spatial derivatives, so their raw index has no meaningful
#      "order" to sum). Threaded through `_polynomial_library`/
#      `_chebyshev_library`/`_chebyshev_to_monomial_matrix` uniformly.
#   4. `step_one(z)` (single-argument, every Jacobian-based regularizer
#      from 183-186) is STILL unaffected -- no `z_prev`/`z_prev2` to give,
#      so both new features fall back to architectural zeros there,
#      unchanged from 189's own convention.
#
# Verified via direct construction + gradient checks (both w_t and w_tt
# active together: term count 15 -> 30 at degree=2/max_term_order=5,
# including the w_t*w_tt cross term; gradients flow through both z_prev
# and z_prev2; output demonstrably depends on z_prev2 when it varies) AND
# a real 3-epoch dry run through the full CLI/training-loop path -- no
# nan, term count 30 confirmed on the resulting checkpoint.
#
# Recipe: IDENTICAL to Section 189 (which is IDENTICAL to 187 + w_t) at
# L=100, PLUS --spectral-poly-time-deriv2. Isolates w_tt as the sole
# additional change on top of 189's own already-run baseline.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section190_second_time_derivative

echo "=== [1/2] Stage 1 ONLY: identical to 189 + NEW --spectral-poly-time-deriv2 (adds w_tt_fd = (w-2*w_prev+w_prev2)/dt_snap^2, 30 terms instead of 22) ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 \
  --spectral-field-kind chebyshev --spectral-poly-degree 2 --spectral-poly-max-term-order 5 \
  --spectral-poly-stable-w-xx-w-xxxx --stable-linear-lr-factor 0.02 \
  --spectral-poly-exclude-nonconservative --spectral-poly-no-constant \
  --spectral-poly-time-deriv --spectral-poly-time-deriv-dt-snap 1.0 --spectral-poly-time-deriv2 \
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

echo "=== [2/2] Diagnostics: fitted PDE coefficients (including w_t/w_tt terms), standalone + multi-IC rollout, Jacobian (one-step and k=10-step composed) ==="
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

print('FINAL Stage-1 checkpoint (epoch 199/200, field_kind=%s, poly_time_deriv=%s, poly_time_deriv2=%s, term count=%d):' % (
    body.field_kind, body.poly_time_deriv, body.poly_time_deriv2, body.poly_coeffs.weight.shape[1]))
n_vars = body.max_order + 1 + (1 if body.poly_time_deriv else 0) + (1 if body.poly_time_deriv2 else 0)
free_indices = body._time_deriv_free_indices
term_indices = _polynomial_term_indices(n_vars, body.poly_degree, body.poly_max_term_order, free_indices)
cheb_weight = body._poly_weight().detach().squeeze(0).clone()
char_k = 2 * math.pi * body.K / body.L
B = _chebyshev_to_monomial_matrix(n_vars, body.poly_degree, body.poly_max_term_order, free_indices)
mono_weight = B @ cheb_weight
names = ['w', 'w_x', 'w_xx', 'w_xxx', 'w_xxxx', 'w_xxxxx', 'w_xxxxxx']
names = names[:body.max_order + 1]
if body.poly_time_deriv: names = names + ['w_t']
if body.poly_time_deriv2: names = names + ['w_tt']
rows = []
for i, idx in enumerate(term_indices):
    if len(idx) == 0:
        label = '(const)'; denom = 1.0
    else:
        label = '*'.join(names[j] for j in idx)
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
print('w_t/w_tt-involving terms specifically:')
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

echo "=== [3/3] Real Lyapunov spectrum (Benettin/QR, full d_latent=96 directions) -- decisive chaos-vs-not check, same as Sections 187/189's ==="
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

echo "=== Section 190 complete ==="
