#!/bin/zsh
# User-directed 2026-09-18: "I would like to try the same thing on L =
# 22." Section 187 (Chebyshev-basis PDE, degree=2/max_term_order=5, soft-
# stable w_xx/w_xxxx, delta_cap 0.5, two-sided + multistep spectrum-shape)
# ran at L=100 and produced a genuinely new failure mode: bounded and
# non-collapsing across a 20-IC rollout (unlike every forced_burgers
# variant 180-186), but the ACTUAL Benettin/QR Lyapunov spectrum showed
# this is NOT chaos -- lambda1~0.00013 (true KS: ~0.086, 660x too weak),
# 40/96 positive exponents (true KS: ~13-15), D_KY=76.3 (true KS: 21-24,
# and the 76.3 figure is itself an artifact of dozens of near-zero
# exponents making the Kaplan-Yorke bracket unreliable). Near-neutral,
# massively degenerate dynamics, not runaway growth and not collapse.
#
# THIS section: identical recipe, only --dataset swapped to the L=22 KS
# regime (stage1_trajectories_L22_dtsnap1.h5, verified to exist: NX=256,
# dt_snap=1.0, same convention as the L=100 dataset). Local_field encoder
# config (n_sites=32, channels=3 -> d_latent=96) and --spectral-K 49/
# --spectral-L 96.0 stay UNCHANGED -- these describe the self-FFT over
# the 96-dim LATENT INDEX itself (d_latent=96 either way, since the
# encoder architecture is not rescaled for the physical domain -- same
# established precedent as Sections 156/157's own L=22 runs), not the
# true physical KS domain length, so they have no L-dependence to update.
#
# ONE genuine adjustment: --spectrum-shape-n-expand and
# --spectrum-shape-expand-target were derived from Section 85's own real
# L=100 Lyapunov spectrum (n_positive=13, lambda1=0.086 -> multiplier
# 1.1). L=22's true KS benchmark is very different (CLAUDE.md's own
# replication targets: lambda1 ~ 0.043-0.05, D_KY ~ 5.2-5.6) -- reusing
# the L=100-calibrated targets unchanged would be asking this run to
# match a MUCH busier spectrum than L=22 KS actually has. Reset to
# n_expand=4 (a reasonable estimate near D_KY~5.4 -- n_positive is
# typically close to the Kaplan-Yorke cutoff) and expand_target=1.05
# (from lambda1~0.05 -> exp(0.05*dt_snap=1.0)~1.051). NOT a precise
# per-mode measurement the way Section 133's L=100 table was (no
# equivalent L=22 per-mode multiplier table was computed this session) --
# a ballpark derived from CLAUDE.md's documented range, flagged here
# explicitly rather than silently reused. contract_floor stays 0.6
# (generic dissipative floor, not domain-specific). Multistep target
# becomes 1.05^10~1.63 automatically (spectrum_shape_multistep_k=10
# unchanged), appropriately more modest than L=100's 2.59.
#
# Caveat worth keeping in mind when reading results: this arc has never
# reduced d_latent for L=22 (96 dims for a system whose true attractor
# dimension here is only ~5.4) -- most of the 96 latent channels are
# "extra" relative to L=22's true intrinsic dimensionality, more so than at
# L=100 (96 dims vs D_KY~21-24). This is an existing, not new, choice
# (matches Sections 156/157 exactly) but affects how any n_positive/D_KY
# result here should be read.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section188_chebyshev_L22
DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5

echo "=== [1/2] Stage 1 ONLY: local_field encoder + spectral_pde_raw propagator, field_kind=chebyshev (degree=2, max_term_order=5, 15 terms), L=22 dataset, spectrum-shape targets rescaled for L=22 (n_expand=4, expand_target=1.05) + delta_cap 0.5 + multistep (k=10), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --encoder local_field \
  --local-field-n-sites 32 --local-field-channels 3 --local-field-mix-radius 2 \
  --local-field-n-mix-layers 3 --local-field-hidden 32 \
  --aux-backbone spectral_pde_raw --mode markovian \
  --spectral-K 49 --spectral-L 96.0 --ode-substeps 1 \
  --spectral-field-kind chebyshev --spectral-poly-degree 2 --spectral-poly-max-term-order 5 \
  --spectral-poly-stable-w-xx-w-xxxx --stable-linear-lr-factor 0.02 \
  --spectral-poly-exclude-nonconservative --spectral-poly-no-constant \
  --spectral-integrator euler \
  --prop-delta-cap 0.5 \
  --w-spectrum-shape 0.4 --spectrum-shape-n-expand 4 --spectrum-shape-expand-target 1.05 --spectrum-shape-contract-floor 0.6 --spectrum-shape-n-samples 32 --spectrum-shape-two-sided \
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

echo "=== [2/2] Diagnostics on the Stage-1-only checkpoint: fitted PDE coefficients (Chebyshev->monomial), standalone rollout, multi-IC rollout (bounded/collapse/blowup check), Jacobian (one-step AND k=10-step composed) ==="
mamba run -n da_env python -c "
import math
import h5py, torch
from torch.func import jacrev
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.propagator import _polynomial_term_indices, _chebyshev_to_monomial_matrix

DATASET = '$DATASET'
ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$AUX')
ae.eval(); prop.eval()
body = prop.body.inner if hasattr(prop.body, 'inner') else prop.body

print('FINAL Stage-1 checkpoint (epoch 199/200, field_kind=%s, degree=%d, max_term_order=%s, L=22 dataset):' % (
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

with h5py.File(DATASET,'r') as f:
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
print('K=10-STEP COMPOSED: targets were 1.05^10=%.3f (top4) / 0.6^10=%.6f (rest)' % (1.05**10, 0.6**10))
" > artifacts/logs/diagnostics_${TAG}.log 2>&1
cat artifacts/logs/diagnostics_${TAG}.log

echo "=== [bonus] Rollout visualization ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$AUX" --dataset "$DATASET" \
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

with h5py.File('$DATASET','r') as f:
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

echo "=== Section 188 complete ==="
