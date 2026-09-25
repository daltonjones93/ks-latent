#!/bin/zsh
# Follow-up to section220_ks_localfield_maskedmlp_logbarrier_k2.sh, user-
# directed 2026-09-24: "sadly looks like stage 2 is blowing up. maybe we
# shouldn't use the log barrier in stage 2." Confirmed: val_kmax_mse
# oscillated wildly and trended into the hundreds of thousands (epoch
# 177: 99152, epoch 181: 165576) -- the barrier's increasingly strong
# gradient near the ceiling appears to inject destabilizing spikes into
# Stage 2's own multi-step unrolled optimization (k-curriculum climbing
# toward 12), a different situation from Stage 1's single-step-per-sample
# evaluation, where the same barrier produced the mildest Stage-1 blowup
# trajectory of any masked_mlp attempt yet.
#
# Reuses Section 220's ALREADY-COMPLETED Stage 1 checkpoint (AE + aux
# propagator) -- no need to rerun Stage 1. Stage 2 this time uses
# Section 52's exact plain warmstart schedule, NO --w-multistep-growth-
# barrier (or any other new regularizer) -- everything else identical.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=section220_ks_localfield_maskedmlp_logbarrier_k2
AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [1/2] Stage 2 (NO growth barrier this time): warm-started from Section 220's Stage 1, Section 52's exact schedule (k_max=12, 300 epochs) ==="
STAGE2_TAG="${TAG}_nobarrier_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

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
print('[$LABEL] (211: D3=0.1883/D_KY=23.04; 216: D3=0.2146/D_KY=22.59; 217: D3=0.2970/diverged; 218: D3=0.3661/diverged; 219: D3=0.4052/diverged; 220 stage1: D3=0.7632/diverged-slowly)')
"
}

echo "=== [2/2] Stage 2 diagnostics ==="
_diagnose "$STAGE2_PROP" "stage2_nobarrier" > artifacts/logs/lyapunov_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/lyapunov_${STAGE2_TAG}.log

echo "=== Section 220b (Stage 2, no barrier) complete ==="
