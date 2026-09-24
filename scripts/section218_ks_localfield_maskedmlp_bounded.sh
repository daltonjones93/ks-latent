#!/bin/zsh
# User-directed 2026-09-24: "weird. I still think there's hope for
# masked_mlp. I just think we need to bound the growth of the
# propagator during stage 1 and we need more rollout of the propagator
# in stage 1, so the encoder can try to compensate to avoid this kind
# of blow up. Please implement these two things and rerun as 218."
#
# Context: Section 217 (local_field + masked_mlp, attn_window=18, no
# D3 loss) diverged catastrophically -- max|z| reached 1e30, crashed
# the Lyapunov computation, and Stage 2 could not rescue it (NaN by
# step ~350). Two new levers, both requested:
#
# 1. "bound the growth of the propagator during stage 1": NEW
#    --w-prop-magnitude-ceiling (ks_latent.training.losses.propagator_
#    rollout_magnitude_ceiling_loss, Section 218) -- ceilings the raw
#    max|z| of aux's own UNSUPERVISED autoregressive rollout during
#    Stage 1, sharing --w-prop-energy-floor's existing rollout
#    machinery (same ramped horizon, same detached starting state --
#    gradient reaches aux's own parameters only, not the encoder).
#    ceiling=15.0 (this project's own measured max|z| scale for
#    genuinely bounded checkpoints, ~9-10).
# 2. "more rollout of the propagator in stage 1, so the encoder can try
#    to compensate": --multistep --k-pred-max 8 (already-existing
#    flags -- ramps the AUXILIARY PROPAGATOR'S OWN L_pred rollout
#    horizon from 2 to 8 over the first 30% of epochs; UNLIKE the
#    magnitude-ceiling term above, gradient from THIS term DOES reach
#    the encoder, since L_pred decodes z_pred back to physical space
#    and compares against real future states -- exactly the "encoder
#    can compensate" mechanism requested).
#    CAUTION, precedent risk worth watching for: Section 203 found
#    --multistep caused STAGE 1 ITSELF to collapse in a DIFFERENT
#    context (ViT encoder + L96 + mode=markovian, no architectural
#    locality) -- a genuinely different combination from this one
#    (masked_mlp + KS + local_field, which already has architectural
#    locality on the encoder side), so not assumed to transfer, but
#    worth explicitly checking Stage-1 D_KY doesn't collapse to 0 here
#    for a different reason than last time.
#
# Otherwise identical to Section 217: local_field encoder (n_sites=32,
# local_channels=3, d_latent=96) + masked_mlp propagator (markovian,
# attn_window=18), Section 52's plain regularizers, L=100/NX=256
# dt_snap=1.0.
#
# Verified via a real (--epochs 2) dry run before this launch: trains
# cleanly with all of --w-prop-magnitude-ceiling/--multistep/
# --k-pred-max together, k ramps to 8 by epoch 1 as expected, no error.
# Full 735-test suite passes (6 new tests for the magnitude-ceiling loss).
#
# Stage 1 AND Stage 2 both run (the standing "always run Stage 2" rule).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=section218_ks_localfield_maskedmlp_bounded

echo "=== [1/4] Stage 1: local_field + masked_mlp propagator (markovian, attn_window=18), Section 52's regularizers PLUS --w-prop-magnitude-ceiling 0.1 (ceiling=15.0) AND --multistep --k-pred-max 8, L=100/NX=256 dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder local_field --aux-backbone masked_mlp --mode markovian \
  --attn-window 18 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --w-prop-magnitude-ceiling 0.1 --prop-magnitude-ceiling-value 15.0 --prop-energy-floor-rollout-k 30 \
  --multistep --k-pred-max 8 \
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
print('[$LABEL] (211: D3=0.1883/D_KY=23.04; 216: D3=0.2146/D_KY=22.59; 217 (masked_mlp, no bound): D3=0.2970/diverged-to-NaN)')
"
}

echo "=== [2/4] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== [3/4] Stage 2: warm-started from Stage 1, Section 52's exact schedule (k_max=12, 300 epochs) ==="
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

echo "=== Section 218 complete ==="
