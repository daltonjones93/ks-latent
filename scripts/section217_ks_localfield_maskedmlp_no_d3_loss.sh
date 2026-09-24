#!/bin/zsh
# User-directed 2026-09-24: "Can we run 211 with the masked mlp instead.
# get rid of the D3 regularizer. I think maybe that could increase
# bandedness as well. this is the last test then we will move on to
# phase b in docs/steps_4-3.md"
#
# Hypothesis: instead of REGULARIZING a global mlp propagator toward
# bandedness (soft, learned, imperfect -- Sections 215/216), use
# --aux-backbone masked_mlp, whose weights are architecturally masked
# to exactly zero outside a fixed circular band (ks_latent.models.
# propagator.MaskedLinear -- "exactly zero gradient outside the band...
# stay EXACTLY zero for the entire time this module is trained," per
# that class's own docstring). For a LINEAR layer the Jacobian IS the
# weight matrix, so a single MaskedLinear's own Jacobian is EXACTLY
# band-limited by construction, not approximately so via a loss
# penalty -- a multi-layer _MaskedMLPDeltaBody's composed Jacobian can
# have a wider effective band (composing k band-limited layers), but
# still has FINITE support, architecturally, regardless of training.
# This may give D3 bandedness "for free" without needing any Jacobian
# regularizer at all, and -- unlike Sections 215/216's soft penalty --
# without competing against chaos/reconstruction for gradient budget.
#
# Exact copy of Section 211's Stage 1 command with --aux-backbone mlp
# swapped for --aux-backbone masked_mlp --attn-window 18 (Section 212's
# own original choice: local_field's own encoder receptive field is
# mix_radius(2)*n_mix_layers(3)=6 sites each direction, ~18 raw
# d_latent units at local_channels=3 -- see Section 212's own header
# for the full derivation). NO --w-jacobian-bandedness/--w-jacobian-
# diagonal-bound this time -- exactly Section 52's plain regularizer
# recipe (w_var/w_spatial/w_logdet only), per this test's own point:
# does the ARCHITECTURE alone induce bandedness, without any Jacobian
# regularizer's help.
#
# Verified via a real (--epochs 2) dry run before this launch: recon
# 0.495 -> 0.026 over 2 epochs (val_recon_final=0.0146), ~4.2s/epoch,
# no error (this exact combination was already dry-run once before,
# Section 212, but never carried through Stage 2 or diagnosed with
# this rigor -- re-verified here before relaunching).
#
# Stage 1 AND Stage 2 both run (the standing "always run Stage 2" rule).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=section217_ks_localfield_maskedmlp_no_d3_loss

echo "=== [1/4] Stage 1: local_field encoder (n_sites=32, local_channels=3, d_latent=96) + LOCAL masked_mlp propagator (markovian, attn_window=18), Section 52's PLAIN regularizers only (no D3 loss), L=100/NX=256 dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder local_field --aux-backbone masked_mlp --mode markovian \
  --attn-window 18 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
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
print('[$LABEL] d_latent:', d_latent, ' backbone:', prop.cfg.backbone, ' attn_window:', prop.cfg.attn_window)
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
n_offdiag_zero = (np.abs(d3.A_jacobian) < 1e-6).sum() - (np.abs(np.diagonal(d3.A_jacobian)) < 1e-6).sum()
print('[$LABEL] exactly-zero off-diagonal Jacobian entries: %d / %d' % (n_offdiag_zero, d_latent*d_latent - d_latent))
print('[$LABEL] (Section 211 (global mlp, no D3 loss): D3=0.1883; Section 216 (global mlp + D3 losses in Stage1): D3=0.2146)')
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

echo "=== Section 217 complete ==="
