#!/bin/zsh
# User-directed 2026-09-18: "I wonder what would happen if we just
# trained this same setup with k=2 for 300 epochs. let's add w_xxxxx
# too, why not."
#
# Motivation: Section 190 (w_t + w_tt on top of 187's Chebyshev PDE)
# showed lambda1 climbing with each added feature (187: 0.00013 -> 189:
# 0.000213 -> 190: 0.000513) but the REAL nonlinear advection w*w_x
# coefficient shrinking toward zero while w*w_t/w_tt-involving terms grew
# to dominate the fit (w*w_t=0.649, the largest term after w_xx/w_xxxx) --
# a real concern that the model is learning to "coast" on its own recent
# rollout history rather than discovering genuine spatial PDE structure.
# One candidate driver: every section since 175 has used a K-CURRICULUM
# ramping k_pred 2->10 over the first ~30% of epochs, then training the
# REMAINDER at k=10 -- meaning the model spends most of its budget being
# pushed to get a hard 10-step-ahead prediction right, which may reward
# exactly this kind of self-referential shortcut (predicting via momentum
# rather than physics) more than genuine short-horizon accuracy would.
# THIS section tests that directly: fix k_pred_max=2 (no curriculum ramp
# at all, k_min==k_max) for the FULL run, quadrupling epoch budget (200->
# 300) so the model gets comparably more actual gradient steps focused
# purely on getting the 1-2 step map right.
#
# Second, independent change: --spectral-max-order 5 (adds w_xxxxx to the
# derivative stack, "why not" -- more expressive power, direct test of
# whether the model can find genuine use for a higher spatial order).
# Requires bumping --spectral-poly-max-term-order 5 -> 6 too: a BARE
# w_xxxxx term has combined order 5, which the existing max_term_order=5
# budget would EXCLUDE (order must be STRICTLY less than max_term_order --
# same off-by-one structure Section 189 hit for w_t, but here it's a
# GENUINE spatial order, not a free_indices-exempt variable, so w_xxxxx
# needs real budget headroom, not an exemption). Verified directly before
# this launch: at max_term_order=6, bare (5,)=w_xxxxx is included; at 5 it
# would not be. n_vars now 8 (w..w_xxxxx=6, w_t, w_tt), term count 36
# (up from 190's 30).
#
# Everything else identical to 190: Chebyshev basis, degree=2,
# soft-stable w_xx/w_xxxx, exclude-nonconservative, no-constant, w_t AND
# w_tt both active, delta_cap 0.5, two-sided + multistep (k=10, UNCHANGED
# from prior sections despite k_pred_max dropping to 2 -- this keeps the
# long-horizon composed-Jacobian diagnostic/regularizer comparable
# apples-to-apples across 187/189/190/191, testing whether genuine
# 10-step behavior improves even without direct 10-step training
# pressure).
#
# Verified via direct construction + gradient check (max_order=5,
# max_term_order=6, w_t+w_tt both on: term count 36, bare w_xxxxx
# confirmed included, rollout finite, gradients flow) before this launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section191_k2_longtrain_5thorder

echo "=== [1/2] Stage 1 ONLY: identical to 190 + --spectral-max-order 5 (adds w_xxxxx) + --spectral-poly-max-term-order 6 (headroom for bare w_xxxxx) + --k-pred-max 2 (no curriculum ramp, fixed at k=2 the whole run) + --epochs 300 ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 --spectral-max-order 5 \
  --spectral-field-kind chebyshev --spectral-poly-degree 2 --spectral-poly-max-term-order 6 \
  --spectral-poly-stable-w-xx-w-xxxx --stable-linear-lr-factor 0.02 \
  --spectral-poly-exclude-nonconservative --spectral-poly-no-constant \
  --spectral-poly-time-deriv --spectral-poly-time-deriv-dt-snap 1.0 --spectral-poly-time-deriv2 \
  --spectral-integrator euler \
  --prop-delta-cap 0.5 \
  --w-spectrum-shape 0.4 --spectrum-shape-n-expand 13 --spectrum-shape-expand-target 1.1 --spectrum-shape-contract-floor 0.6 --spectrum-shape-n-samples 32 --spectrum-shape-two-sided \
  --w-spectrum-shape-multistep 0.4 --spectrum-shape-multistep-k 10 --spectrum-shape-multistep-n-samples 16 \
  --k-pred-max 2 \
  --w-decorr 0 --w-var 0.01 --w-var-floor 0 --w-spatial 0.12 --spatial-signed --w-logdet 0.01 --w-smooth 0.012 \
  --w-channel-mean 0.01 \
  --full-propagator --amp \
  --epochs 300 --checkpoint-every 30 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/2] Diagnostics: fitted PDE coefficients (including w_xxxxx/w_t/w_tt terms), standalone + multi-IC rollout, Jacobian (one-step and k=10-step composed) ==="
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

print('FINAL Stage-1 checkpoint (epoch 299/300, field_kind=%s, max_order=%d, poly_time_deriv=%s, poly_time_deriv2=%s, term count=%d):' % (
    body.field_kind, body.max_order, body.poly_time_deriv, body.poly_time_deriv2, body.poly_coeffs.weight.shape[1]))
n_vars = body.max_order + 1 + (1 if body.poly_time_deriv else 0) + (1 if body.poly_time_deriv2 else 0)
free_indices = body._time_deriv_free_indices
term_indices = _polynomial_term_indices(n_vars, body.poly_degree, body.poly_max_term_order, free_indices)
cheb_weight = body._poly_weight().detach().squeeze(0).clone()
char_k = 2 * math.pi * body.K / body.L
B = _chebyshev_to_monomial_matrix(n_vars, body.poly_degree, body.poly_max_term_order, free_indices)
mono_weight = B @ cheb_weight
names = ['w', 'w_x', 'w_xx', 'w_xxx', 'w_xxxx', 'w_xxxxx', 'w_xxxxxx'][:body.max_order + 1]
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
print('w_xxxxx-involving terms specifically:')
for label, c, m in rows:
    if 'w_xxxxx' in label:
        print(' ', label.ljust(20), ('%.6f' % c).rjust(16), ('%.6f' % m).rjust(26))
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

echo "=== [3/3] Real Lyapunov spectrum (Benettin/QR, full d_latent=96 directions) -- decisive chaos-vs-not check, same as Sections 187/189/190's ==="
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

echo "=== Section 191 complete ==="
