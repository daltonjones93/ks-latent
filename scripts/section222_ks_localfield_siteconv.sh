#!/bin/zsh
# User-directed 2026-09-24: "could we use a model like the encoder from
# 221 as a propagator? I've never thought of trying that" -- built
# backbone="site_conv" (_SiteConvDeltaBody, ks_latent/models/propagator.py):
# reuses KSAutoencoderLocalField's own circular-Conv1d site-mixing design
# (1x1 conv local_channels->H, n_layers circular convs mixing across
# sites, 1x1 conv H->local_channels, bias=False on the output projection)
# as a dimension-preserving z->z map on the latent's native (n_sites,
# local_channels) grid, instead of a masked dense layer or a flat-sequence
# conv. H=32/n_layers=3 default to LocalFieldAutoencoderConfig's own
# already-validated values; attn_window (reused) = the site-mixing
# radius, set to 2 here matching the encoder's own site_mix_radius
# default (effective receptive field = n_layers*radius = 6 sites,
# verified via test_propagator_site_conv.py's own Jacobian measurement).
#
# User: "let's try this after 221... launch 222 with the new propagator."
# Clean first test: Section 211/217's exact plain baseline recipe
# (Section 52 regularizers only -- w_decorr=0, w_var=0.02, w_spatial=0.01
# --spatial-signed, w_var-floor=0, w_logdet=0.0035, --full-propagator,
# --amp; plain fixed k_pred=2, no --multistep), NO D3 loss, NO growth-
# barrier -- this is a genuinely new architecture (circular convs on the
# true site grid, not a masked dense layer), so testing it against the
# same clean baseline masked_mlp was first tested against (Section 217)
# gives an uncofounded read before adding any of the levers built for
# masked_mlp's specific instability.
#
# Verified via real (--epochs 2) dry runs of BOTH stages already (before
# this script was written): trains cleanly end to end, Stage 2's
# val_kmax_mse dropped to 0.11 after just 2 epochs. Full 785-test suite
# passes (11 new tests for site_conv: shape, identity-at-init, no-bias,
# receptive field, circular wraparound, gradient flow, config
# validation).
#
# Otherwise identical dataset/domain to every other section this
# session: local_field encoder (n_sites=32, local_channels=3,
# d_latent=96), L=100/NX=256 dt_snap=1.0.
#
# Stage 1 AND Stage 2 both run (the standing "always run Stage 2" rule).
# Lyapunov RuntimeError wrapped in try/except (Section 217's crash fix).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=section222_ks_localfield_siteconv

echo "=== [1/4] Stage 1: local_field + site_conv propagator (markovian, attn_window=2 -> effective receptive field 6 sites), Section 52's regularizers only, plain k_pred=2, L=100/NX=256 dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder local_field --aux-backbone site_conv --mode markovian \
  --attn-window 2 \
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
print('[$LABEL] (211: D3=0.1883/D_KY=23.04; 216: D3=0.2146/D_KY=22.59; 221: D3=0.3916/D_KY=22.68 [masked_mlp_expand, site radius 4])')
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

echo "=== Section 222 complete ==="
