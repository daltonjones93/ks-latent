#!/bin/zsh
# User-directed 2026-09-24: "it looks like stage 2 is having trouble
# converging. If we combined the D3 regularizer with a term that
# bounded the magnitude of the diagonal of the jacobian, maybe that
# would help. kill the current stage 2 run, design this regularizer,
# and run this as stage 216." Then, separately (standing instruction,
# saved to memory): "stage 2 should be run by default for all runs."
#
# Context: Section 215 (--w-jacobian-bandedness 0.05 alone) pushed D3's
# own bandedness score from Section 211's baseline 0.19 to 0.99 (near-
# perfect) but made standalone-rollout divergence WORSE, not better:
# D_KY 74.93 -> 90.64, max|z| reaching 6265 (vs 211's 1019) over the
# same 2000 steps. Diagnosis: bandedness constrains WHERE coupling
# concentrates, not HOW LARGE the surviving (near-diagonal, including
# self-coupling) entries are -- forcing locality apparently just
# concentrated the instability into sharper per-site growth.
#
# NEW: --w-jacobian-diagonal-bound (ks_latent.training.losses.
# propagator_jacobian_diagonal_bound_loss, Section 216) -- caps each
# site's own self-coupling magnitude (the diagonal of the propagator's
# step-Jacobian) directly, orthogonal to bandedness (verified via a
# dedicated unit test that two Jacobians with identical diagonals but
# very different off-diagonal structure score IDENTICALLY under this
# loss). Shares the same Jacobian computation helper
# (_propagator_step_jacobian_matrix) as the bandedness loss, factored
# out for this pairing specifically. Ceiling=1.5 reuses --spectrum-
# shape-expand-target's own calibration point (Section 85) -- a first
# attempt, not independently tuned for the diagonal.
#
# Verified via a real (--epochs 2) dry run before this launch: both
# regularizers active together, trains cleanly, recon 0.488 -> 0.026
# over 2 epochs, no error. Full 729-test suite passes (6 new tests for
# the diagonal-bound loss).
#
# Stage 1 AND Stage 2 both run this time (the standing "always run
# Stage 2" rule, learned directly from Section 211/213/215's own
# pattern: Stage 1 alone reliably overshoots into apparent instability,
# only Stage 2's own standalone rollout is the real bar).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=section216_ks_localfield_jacobian_bandedness_diagonal_bound

echo "=== [1/4] Stage 1: local_field encoder (n_sites=32, local_channels=3, d_latent=96) + plain MLP propagator (markovian), Section 52's regularizers PLUS --w-jacobian-bandedness 0.05 AND --w-jacobian-diagonal-bound 0.1 (ceiling=1.5), L=100/NX=256 dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder local_field --aux-backbone mlp --mode markovian \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --w-jacobian-bandedness 0.05 --jacobian-bandedness-bandwidth 3.0 --jacobian-bandedness-n-samples 32 \
  --w-jacobian-diagonal-bound 0.1 --jacobian-diagonal-bound-ceiling 1.5 --jacobian-diagonal-bound-n-samples 32 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

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
print('[$LABEL] d_latent:', d_latent)
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
res = lyapunov_spectrum_latent_propagator(
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=2000, qr_every=10,
    dt_snap=1.0, warmup_steps=200, seed=0, max_abs_state=1e4,
)
print('[$LABEL] lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('[$LABEL] D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('[$LABEL] D_KY: could not bracket --', e)
print('[$LABEL] (true L100 target: D_KY in [21,24])')

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
diag_mag = np.abs(np.diagonal(d3.A_jacobian))
print('[$LABEL] D3 Jacobian diagonal |magnitude|: mean=%.4f  max=%.4f  frac>1.5=%.4f' % (
    diag_mag.mean(), diag_mag.max(), (diag_mag > 1.5).mean()))
print('[$LABEL] (Section 211 baseline: D3=0.1883; Section 215 (bandedness alone): D3=0.9938, D_KY=90.64, max|z|@1999=6265)')
"
}

echo "=== [2/4] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== [3/4] Stage 2: warm-started from Stage 1, Section 52's exact schedule (k_max=12, 300 epochs) -- per the 'always run Stage 2' rule ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [4/4] Stage 2 diagnostics ==="
_diagnose "$STAGE2_PROP" "stage2" > artifacts/logs/lyapunov_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/lyapunov_${STAGE2_TAG}.log

echo "=== Section 216 complete ==="
