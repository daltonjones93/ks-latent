#!/bin/zsh
# User-directed 2026-09-21, following Section 192's result. 192 added the
# REAL-DATA-anchored two-sided + multistep spectrum-shape regularizers to
# pde_head alone (aux untouched, confirmed: aux stayed genuinely chaotic,
# D_KY=22.34, essentially matching 167's own 22.67). But pde_head got
# WORSE, not better: standalone rollout now DIVERGES (first non-finite
# step 131, a regression -- 167's own pde_head never diverged, it just
# decayed smoothly to 0.275 by t=199); the real Lyapunov spectrum on a
# different real IC that stayed bounded showed n_positive=0/96, D_KY=0.0
# -- every exponent negative, a PURER collapse than before.
#
# Diagnosis (discussed before launching 192, now directly confirmed by
# its result): w_pde_spectrum_shape/w_pde_spectrum_shape_multistep are
# evaluated ONLY on real, encoder-derived on-attractor states -- they
# never see what pde_head's OWN free rollout actually visits. Training
# itself only ever exposed pde_head to a 4-step real-rollout distillation
# window (--pde-distill-real-rollout-k 4); step-131 divergence was never
# in view during training at all. User: "launch it now" (the self-
# rollout-sampled version built and verified alongside 192's own launch,
# ks_latent/training/loops.py's new `_pde_head_self_rollout_states`
# helper + w_pde_spectrum_shape_self/w_pde_spectrum_shape_multistep_self,
# Stage1/2TrainingConfig fields, CLI flags on both train_stage{1,2}_
# patched.py -- all added and syntax/wiring-verified, 5-epoch Stage 2
# smoke test run clean with both new flags active, before this launch).
#
# THIS section: Stage 1 is IDENTICAL to 192 (and 167) -- UNCHANGED, so
# reused directly from 192's own Stage-1 checkpoints rather than
# retraining (192's Stage 1 never touched pde_head's spectrum-shape
# regularizers at all -- those are Stage-2-only in both sections -- so
# this is a legitimate, exact warm-start, not an approximation, and
# isolates Stage 2's regularizer change as the sole variable versus 192).
# Stage 2: IDENTICAL to 192's own Stage 2 (same real-data spectrum-shape
# terms KEPT, not replaced -- see Section 192's own launch script for the
# reasoning: the real-data anchoring is presumably still doing useful
# work near the attractor) PLUS NEW --w-pde-spectrum-shape-self 0.4
# --w-pde-spectrum-shape-multistep-self 0.4 (same n_expand/expand_target/
# contract_floor/two_sided targets, shared with the real-data version),
# --pde-spectrum-shape-self-rollout-k 20 (roll pde_head forward 20 steps
# under no_grad from real starting states before sampling the evaluation
# state -- see w_pde_spectrum_shape_self's own docstring in config.py for
# the full mechanism).
set -e
cd /Users/daltonjones/Documents/latent_DA

BASE_TAG=section192_pdehead_spectrum_shape
TAG=section193_pdehead_selfrollout_spectrum_shape

AE=artifacts/stage1_ae_patched_full_${BASE_TAG}.pt
AUX=artifacts/stage1_prop_full_${BASE_TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${BASE_TAG}.pt

echo "=== [1/3] Stage 2: warm-started from Section 192's own (unchanged) Stage 1 checkpoints. IDENTICAL to 192's Stage 2 (real-data spectrum-shape terms kept) PLUS NEW --w-pde-spectrum-shape-self 0.4 --w-pde-spectrum-shape-multistep-self 0.4 (self-rollout-sampled, pde_head.step_one ONLY, aux untouched), k_max=12, 300 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --init-pdehead-checkpoint "$PDEHEAD_STAGE1" \
  --w-pde-distill-real 0.5 \
  --w-pde-distill-real-rollout 0.3 --pde-distill-real-rollout-k 4 \
  --w-pde-nonlinear-l2 0.01 \
  --w-pde-spectrum-shape 0.4 --pde-spectrum-shape-n-expand 13 --pde-spectrum-shape-expand-target 1.1 --pde-spectrum-shape-contract-floor 0.6 --pde-spectrum-shape-n-samples 32 --pde-spectrum-shape-two-sided \
  --w-pde-spectrum-shape-multistep 0.4 --pde-spectrum-shape-multistep-k 10 --pde-spectrum-shape-multistep-n-samples 16 \
  --w-pde-spectrum-shape-self 0.4 --pde-spectrum-shape-self-rollout-k 20 --pde-spectrum-shape-self-n-samples 16 \
  --w-pde-spectrum-shape-multistep-self 0.4 \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt
STAGE2_PDEHEAD=artifacts/stage2_pdehead_patched_full_${STAGE2_TAG}.pt

echo "=== [2/3] pde_head standalone rollout (direct comparison: 167 never diverged/decayed to 0.275; 192 diverged at step 131 -- did the self-rollout term fix it?) ==="
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
print('pde_head standalone rollout (200 steps) first non-finite step (-1=never):', first_bad)
for t in [0,10,20,40,60,80,100,131,150,199]:
    v = z_roll[0,t]
    print(' t=%d max|z|=%s' % (t, v.abs().max().item() if torch.isfinite(v).all() else 'nan/inf'))
print('(167: 1.818->1.889->...->0.958(t=80)->0.275(t=199), decaying, never diverges)')
print('(192: diverged at step 131)')

# multi-IC check too, same convention as the spectral_pde_raw arc's own diagnostics
with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj20 = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj20.shape
with torch.no_grad():
    z_all = ae.encode(traj20.reshape(n*T, NX)).reshape(n, T, -1)
z0b = z_all[:,0,:]
with torch.no_grad():
    traj_roll = pdehead.rollout(z0b, z0b, k=150)
finite20 = torch.isfinite(traj_roll).all(dim=(0,2))
first_bad20 = int((~finite20).float().argmax().item()) if (~finite20).any() else -1
print()
print('multi-IC (20) rollout first non-finite step:', first_bad20)
if first_bad20 == -1:
    maxz_per_t = traj_roll.abs().amax(dim=(0,2))
    for t in [0,5,10,20,40,60,90,120,149]:
        print(' t=%d  max|z| across all 20 ICs: %.4f   per-IC max|z| range: [%.4f, %.4f]' % (
            t, maxz_per_t[t].item(),
            traj_roll[:,t,:].abs().amax(dim=-1).min().item(),
            traj_roll[:,t,:].abs().amax(dim=-1).max().item(),
        ))
"

echo "=== [3/3] Real Lyapunov spectrum for BOTH aux (sanity check: still ~22.3-22.7, untouched again) and pde_head (the actual test) ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
aux, aux_cfg, _ = load_propagator_checkpoint('$STAGE2_PROP')
pdehead, pdehead_cfg, _ = load_propagator_checkpoint('$STAGE2_PDEHEAD')
ae.eval(); aux.eval(); pdehead.eval()
d_latent = aux.cfg.d_latent

with h5py.File('artifacts/datasets/stage1_trajectories_dtsnap1.h5','r') as f:
    traj = torch.tensor(f['trajectories'][3], dtype=torch.float32)
with torch.no_grad():
    z01 = ae.encode(traj[:2]).numpy()

print('--- aux propagator (masked_mlp_expand, should be ~unchanged: 167=22.67, 192=22.34) ---')
res_aux = lyapunov_spectrum_latent_propagator(
    aux, z01, mode='single_state', n_directions=d_latent, n_steps=400, qr_every=5,
    dt_snap=1.0, warmup_steps=50, seed=0, max_abs_state=1e3,
)
print('lambda1:', res_aux.exponents[0], ' n_positive:', res_aux.n_positive, '/', res_aux.n_directions)
try:
    print('D_KY:', res_aux.kaplan_yorke_dimension)
except Exception as e:
    print('D_KY: could not bracket --', e)

print()
print('--- pde_head (the actual test: did the self-rollout term produce a genuine, non-collapsing, non-diverging spectrum?) ---')
res_pde = lyapunov_spectrum_latent_propagator(
    pdehead, z01, mode='single_state', n_directions=d_latent, n_steps=400, qr_every=5,
    dt_snap=1.0, warmup_steps=50, seed=0, max_abs_state=1e3,
)
print('lambda1:', res_pde.exponents[0], ' n_positive:', res_pde.n_positive, '/', res_pde.n_directions)
try:
    print('D_KY:', res_pde.kaplan_yorke_dimension)
except Exception as e:
    print('D_KY: could not bracket --', e)
print('spectrum (first 20):', [round(x,4) for x in res_pde.exponents[:20]])
print('spectrum (last 10):', [round(x,4) for x in res_pde.exponents[-10:]])
print('(192 comparison: lambda1=-0.0055, n_positive=0/96, D_KY=0.0)')
" > artifacts/logs/lyapunov_${TAG}.log 2>&1
cat artifacts/logs/lyapunov_${TAG}.log

echo "=== Section 193 complete ==="
