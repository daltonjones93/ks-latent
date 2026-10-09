#!/bin/zsh
# User-directed 2026-09-27: "sounds good, my last question. Is there any
# value in trying to model the pde using a rational function instead of a
# polynomial?" then: "let's try the rational function idea. keep the
# functional form pretty simple. at most degree 3 in denomenator and
# numerator and try to explicitly avoid poles. Then try 225 again as 226
# with this."
#
# Section 225's exact recipe (local_field n_sites=16/local_channels=3,
# local_mlp propagator, --pde-distill, w_pde_distill=0.2 both stages,
# --pde-poly-degree 2, --pde-integrator euler, --w-pde-spectrum-shape 0.1,
# --w-pde-energy-floor 0.1) with ONLY the pde_head's field_kind swapped:
# chebyshev -> rational (new `_SpectralPDEDeltaBody` branch, see
# ks_latent/models/propagator.py's "rational" docstring and
# ks_latent/config.py's `_validate_spectral_field_kind`, added this
# session specifically for this run).
#
# `field_kind="rational"`: `correction = N(library) / (1 + softplus(D(library)))`
# over the SAME degree-<=poly_degree monomial library polynomial/chebyshev
# use, N and D two separate plain linear layers, N zero-initialized so the
# whole body starts at identity. `1 + softplus(raw) >= 1` for ANY real
# `raw` -- an architectural, provably-always-true pole-avoidance guarantee
# (verified in tests/unit/test_propagator_spectral_pde_rational.py),
# unlike chebyshev's basis-conditioning argument which was never an
# architectural boundedness guarantee to begin with (see Section 225's own
# script header). It does NOT by itself bound the overall magnitude of
# N/D -- that remains the job of --w-pde-spectrum-shape/--w-pde-energy-floor,
# unchanged from 225.
#
# poly_degree kept at 2 (not bumped to the user's stated ceiling of 3):
# the user's "at most degree 3" set an upper bound on the functional form,
# not a mandate to use it, and keeping degree=2 makes this the cleanest
# possible single-variable comparison against Section 225's chebyshev run
# (same poly_degree, only field_kind differs).
#
# --pde-poly-no-constant / --pde-poly-exclude-nonconservative (used in
# 225) are DELIBERATELY DROPPED here, not just inert: they are read only
# by the polynomial/chebyshev branches (never wired into the new
# "rational" branch, confirmed by reading propagator.py's field() method)
# and passing them would silently do nothing -- omitted for clarity, per
# the user's own "keep the functional form pretty simple" instruction,
# which this run interprets as also meaning "don't stack on the
# polynomial-specific stability constraints from earlier attempts."
#
# Everything else (Stage 1 detached / no --pde-mutual, Stage 2
# unconditionally mutual per Stage2TrainingConfig's own design, --amp,
# --full-propagator, Section 52's regularizers, k_max=12/300-epoch Stage 2
# schedule) is IDENTICAL to Section 225 -- see that script's own header
# for the full reasoning, not repeated here.
#
# Dry-run both stages (--epochs 2) before this launch, matching this
# project's own convention for every new architecture/mechanism.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=section226_ks_localfield_localmlp_pdedistill_rational

echo "=== [1/4] Stage 1: Section 224's exact recipe + --pde-distill (DETACHED, rational, degree 2, euler, spectrum-shape + energy-floor), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder local_field --local-field-n-sites 16 --local-field-channels 3 \
  --aux-backbone local_mlp --mode markovian --aux-n-tokens 16 --attn-window 3 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --pde-distill --w-pde-distill 0.2 \
  --pde-field-kind rational --pde-poly-degree 2 --pde-integrator euler \
  --w-pde-spectrum-shape 0.1 --w-pde-energy-floor 0.1 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD_STAGE1=artifacts/stage1_pdehead_full_${TAG}.pt

_diagnose() {
  local PROP_PATH=$1
  local LABEL=$2
  mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.analysis.diagnostics import coupling_graph_diagnostic

DATASET = '$DATASET'
ae, ae_cfg, ae_ckpt = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, prop_ckpt = load_propagator_checkpoint('$PROP_PATH')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('[$LABEL] d_latent:', d_latent, ' backbone:', prop.cfg.backbone)
print('[$LABEL] val_recon_final (from AE checkpoint):', ae_ckpt.get('val_recon_final'))

with h5py.File(DATASET,'r') as f:
    trajectories = torch.tensor(f['trajectories'][:60], dtype=torch.float32)
    traj_val = trajectories[50:60]
    traj3 = trajectories[53]

with torch.no_grad():
    u_hat, z_all = ae(traj_val.reshape(-1, 256))
    recon_mse = ((u_hat - traj_val.reshape(-1,256))**2).mean().item()
print('[$LABEL] held-out recon MSE:', recon_mse)

with torch.no_grad():
    z20 = ae.encode(traj_val.reshape(-1,256)).reshape(10, 251, d_latent)
z0b = z20[:,0,:]
with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=2000)
finite = torch.isfinite(traj_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('[$LABEL] multi-IC (10) rollout first non-finite step:', first_bad)
maxz_per_t = traj_roll.abs().amax(dim=(0,2))
for t in [0,50,100,300,600,1000,1500,1999]:
    if t < traj_roll.shape[1]:
        print('[$LABEL] t=%d  max|z| across all 10 ICs: %.4f' % (t, maxz_per_t[t].item()))
final_states = traj_roll[:,-1,:]
pairwise = torch.cdist(final_states, final_states)
print('[$LABEL] final-state pairwise distance (min excl. diag, max, mean):',
      (pairwise + torch.eye(10)*1e9).min().item(), pairwise.max().item(), pairwise.mean().item())

with torch.no_grad():
    z01 = ae.encode(traj3[:2]).numpy()
try:
    res = lyapunov_spectrum_latent_propagator(
        prop, z01, mode='single_state', n_directions=d_latent, n_steps=2000, qr_every=10,
        dt_snap=1.0, warmup_steps=200, seed=0, max_abs_state=1e4,
    )
    print('[$LABEL] lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
    try:
        print('[$LABEL] D_KY:', res.kaplan_yorke_dimension)
    except Exception as e:
        print('[$LABEL] D_KY: could not bracket --', e)
except RuntimeError as e:
    print('[$LABEL] Lyapunov computation FAILED (reference trajectory diverged):', e)
print('[$LABEL] (true L100 target: D_KY in [21,24]; 224 own: D_KY=22.14, D3=0.4313, bounded; 225 chebyshev: D_KY=21.58, D3=0.4136)')

import numpy as np
rng = np.random.default_rng(0)
n_runs, T, NX = trajectories.shape
run_idx = rng.integers(0, n_runs, size=150)
t_idx = rng.integers(0, T - 1, size=150)
with torch.no_grad():
    u_prev = trajectories[run_idx, t_idx]
    u_curr = trajectories[run_idx, t_idx + 1]
    z_prev = ae.encode(u_prev)
    z_curr = ae.encode(u_curr)
d3 = coupling_graph_diagnostic(prop.step, z_prev, z_curr, n_null=1000, seed=0)
print('[$LABEL] D3 bandedness_observed:', d3.bandedness_observed, ' p_value:', d3.bandedness_p_value)
"
}

_diagnose_pdehead() {
  local PDEHEAD_PATH=$1
  local LABEL=$2
  mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

DATASET = '$DATASET'
ae, ae_cfg, ae_ckpt = load_autoencoder_checkpoint('$AE')
pde_head, pde_cfg, _ = load_propagator_checkpoint('$PDEHEAD_PATH')
ae.eval(); pde_head.eval()
d_latent = pde_cfg.d_latent
print('[$LABEL pde_head] backbone:', pde_cfg.backbone, ' field_kind:', pde_cfg.spectral_field_kind)

with h5py.File(DATASET,'r') as f:
    trajectories = torch.tensor(f['trajectories'][:60], dtype=torch.float32)
    traj_val = trajectories[50:60]

with torch.no_grad():
    z20 = ae.encode(traj_val.reshape(-1,256)).reshape(10, 251, d_latent)
z0b = z20[:,0,:]
with torch.no_grad():
    traj_roll = pde_head.rollout(z0b, z0b, k=2000)
finite = torch.isfinite(traj_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('[$LABEL pde_head] multi-IC (10) OWN standalone rollout first non-finite step:', first_bad)
maxz_per_t = traj_roll.abs().amax(dim=(0,2))
for t in [0,50,100,300,600,1000,1500,1999]:
    if t < traj_roll.shape[1]:
        print('[$LABEL pde_head] t=%d  max|z| across all 10 ICs: %.4f' % (t, maxz_per_t[t].item()))

# distillation-target-tracking quality: pde_head.step_one vs real encoded one-step deltas
with torch.no_grad():
    z_prev = z20[:, :-1, :].reshape(-1, d_latent)
    z_curr = z20[:, 1:, :].reshape(-1, d_latent)
    z_pred = pde_head.step_one(z_prev)
    ss_res = ((z_pred - z_curr) ** 2).sum().item()
    ss_tot = ((z_curr - z_curr.mean(dim=0, keepdim=True)) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot
print('[$LABEL pde_head] one-step R^2 vs real encoded trajectory (held-out):', r2)
print('[$LABEL pde_head] (225 chebyshev own: one-step R^2=0.9226, standalone rollout NaN by step 81)')
"
}

echo "=== [2/4] Stage 1 diagnostics (propagator + pde_head) ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log
_diagnose_pdehead "$PDEHEAD_STAGE1" "stage1" > artifacts/logs/pdehead_${TAG}_stage1.log 2>&1
cat artifacts/logs/pdehead_${TAG}_stage1.log

echo "=== [3/4] Stage 2: warm-started, propagator refined AND pde_head continues (unconditionally mutual, per Stage2TrainingConfig's own design), Section 52's exact schedule (k_max=12, 300 epochs) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --init-pdehead-checkpoint "$PDEHEAD_STAGE1" \
  --w-pde-distill 0.2 --w-pde-spectrum-shape 0.1 --w-pde-energy-floor 0.1 \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt
STAGE2_PDEHEAD=artifacts/stage2_pdehead_patched_full_${STAGE2_TAG}.pt

echo "=== [4/4] Stage 2 diagnostics (propagator + pde_head) ==="
_diagnose "$STAGE2_PROP" "stage2" > artifacts/logs/lyapunov_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/lyapunov_${STAGE2_TAG}.log
_diagnose_pdehead "$STAGE2_PDEHEAD" "stage2" > artifacts/logs/pdehead_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/pdehead_${STAGE2_TAG}.log

echo "=== Section 226 complete ==="
