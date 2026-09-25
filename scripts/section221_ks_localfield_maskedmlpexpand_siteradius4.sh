#!/bin/zsh
# User-directed 2026-09-24, follow-up after Section 220's mixed result
# (Stage 1 with the log-barrier produced the mildest masked_mlp blowup
# trajectory yet, but the direct DA sanity check showed its basic
# forecast accuracy was still 5-8x worse than Section 216 even at a
# single step -- see docs/RESULTS.md's Section 220 writeup). User asked
# what attn_window was set to, then: "right, so let's expand the
# physical site radius to 4, let's use the other masked mlp that
# expands in the second layer, and let's try running 211 again. please
# leave the log penalty out of phase 2" (interpreted as "Stage 2" --
# matches the immediately-preceding finding that the log-barrier loss
# caused catastrophic Stage-2 instability when applied there).
#
# Three changes from Section 220, all requested:
#
# 1. --aux-backbone masked_mlp_expand (was masked_mlp) -- the OTHER
#    masked-mlp variant (_MaskedMLPExpandDeltaBody, Section 128): three
#    MaskedLinearRect layers (input_proj: d_latent->hidden "expand",
#    mid_proj: hidden->hidden "the middle layer" stays widened,
#    output_proj: hidden->d_latent "contract"), hidden=expand_factor*
#    d_latent (--masked-mlp-expand-factor, default 3, kept). Distinct
#    from Section 217/218/219/220's plain masked_mlp (_MaskedMLPDeltaBody,
#    fully dimension-preserving throughout, no expansion anywhere).
# 2. --attn-window 4 (was 9) -- "expand the physical site radius to 4"
#    (up from Section 219/220's site radius 3). IMPORTANT: attn_window's
#    unit conversion is DIFFERENT for masked_mlp_expand than plain
#    masked_mlp -- verified directly via a Jacobian receptive-field
#    measurement (_MaskedMLPExpandDeltaBody with expand_factor=3):
#    window=3 -> site radius 3, window=4 -> site radius 4, window=5 ->
#    site radius 5, a clean 1:1 mapping (unlike plain masked_mlp, where
#    window=9 -> site radius 3, i.e. window = 3*site_radius via the
#    local_channels=3 scaling). So attn_window=4 here, NOT 12 (which
#    would have been the naive site_radius*channels extrapolation from
#    plain masked_mlp's own convention, and would have given a much
#    wider effective radius than intended).
# 3. Otherwise repeats Section 211's own exact recipe (Section 52
#    regularizers only: w_decorr=0, w_var=0.02, w_spatial=0.01
#    --spatial-signed, w_var-floor=0, w_logdet=0.0035, --full-propagator,
#    --amp) -- Section 211 originally used --aux-backbone mlp (global,
#    dense); this run swaps in masked_mlp_expand, same as Section 217's
#    own framing ("run 211 with the masked mlp instead").
#
# --w-multistep-growth-barrier KEPT in STAGE 1 ONLY (k=10, ceiling=75.0,
# weight=0.1, same Section 216 calibration as Section 220) -- produced
# the calmest Stage-1 blowup trajectory of any masked_mlp attempt so far
# there. EXPLICITLY DROPPED from Stage 2 ("leave the log penalty out of
# phase 2") -- Section 220 found it caused catastrophic, oscillating
# Stage-2 instability (val_kmax_mse spiking into the hundreds of
# thousands) when applied to Stage 2's own multi-step unrolled
# optimization, a fundamentally different situation from Stage 1's
# single-step-per-sample evaluation.
#
# Stage 1 stays at plain, fixed k_pred=2 (no --multistep) -- carried
# over from Section 220 ("rollout doesn't seem to help"), not revisited
# this round.
#
# Verified via real (--epochs 2) dry runs of BOTH stages before this
# launch: trains cleanly with masked_mlp_expand + attn_window=4 +
# Stage-1-only growth barrier, no error.
#
# Stage 1 AND Stage 2 both run (the standing "always run Stage 2" rule).
# Lyapunov RuntimeError wrapped in try/except (Section 217's crash fix).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=section221_ks_localfield_maskedmlpexpand_siteradius4

echo "=== [1/4] Stage 1: local_field + masked_mlp_expand propagator (markovian, attn_window=4 = site radius 4, expand_factor=3), Section 52's regularizers PLUS --w-multistep-growth-barrier 0.1 (k=10, ceiling=75.0), plain k_pred=2, L=100/NX=256 dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder local_field --aux-backbone masked_mlp_expand --mode markovian \
  --attn-window 4 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --w-multistep-growth-barrier 0.1 --multistep-growth-barrier-k 10 --multistep-growth-barrier-ceiling 75.0 \
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
print('[$LABEL] (211: D3=0.1883/D_KY=23.04; 216: D3=0.2146/D_KY=22.59; 217: D3=0.2970/diverged; 218: D3=0.3661/diverged; 219: D3=0.4052/diverged; 220 stage1: D3=0.7632/diverged-slowly)')
"
}

echo "=== [2/4] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== [3/4] Stage 2: warm-started from Stage 1, Section 52's exact schedule (k_max=12, 300 epochs), NO growth barrier (per 'leave the log penalty out of phase 2') ==="
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

echo "=== Section 221 complete ==="
