#!/bin/zsh
# User-directed 2026-09-24, follow-up to Section 219 (whose squared-hinge
# growth ceiling + narrowed attn_window diverged FASTER than Section 218's
# unconstrained baseline -- see docs/RESULTS.md's Section 219 writeup).
# Asked "any ideas to make the masked mlp work?" -- explained local_mlp
# (untried, different architecture family) as a Phase F alternative, then
# was given this exact direction for one more masked_mlp attempt:
# "First tell me what the local_mlp option is. Then I think we need to
# lower the 150 bound and penalize this differently. instead of using
# mse, use some kind of -log loss such that there is a boundary at
# wherever we want to bound the jacobian. go back to just training in
# stage 1 with k=2. rollout doesn't seem to help. We also want to
# regularize explosive growth during stage 2 too. keep everything else
# the same."
#
# Three changes from Section 219, all requested:
#
# 1. NEW --w-multistep-growth-barrier (ks_latent.training.losses.
#    propagator_multistep_growth_barrier_loss, Section 220) REPLACES
#    --w-multistep-growth-ceiling. Same underlying quantity (composed
#    10-step Jacobian's top singular value) but a safeguarded LOG-BARRIER
#    penalty shape instead of a squared hinge: the hinge has EXACTLY ZERO
#    gradient anywhere below the ceiling (confirmed directly in
#    test_losses_multistep_growth_barrier.py's own comparison test) --
#    it never discourages approach, only reacts after the boundary is
#    crossed, which is presumably why 219's version didn't help. The
#    barrier grows toward infinity as growth approaches `ceiling` from
#    below (true interior-point-style repulsion), with a safeguarded
#    linear extrapolation beyond the ceiling so training can't produce
#    NaN/inf if a step ever overshoots. `ceiling=75.0` (LOWERED from
#    219's 150.0, ~1.13x Section 216's own observed composed-10-step max
#    of 66.4) -- justified tighter here specifically because a barrier
#    doesn't need the same headroom a reactive hinge does; the closer the
#    ceiling sits to the genuinely-healthy scale, the earlier and
#    stronger the repulsive pressure engages.
# 2. --w-multistep-growth-barrier is ALSO enabled in STAGE 2 this time
#    (new CLI wiring on train_stage2_patched.py, same weight/k/ceiling as
#    Stage 1) -- Sections 218/219 only ever regularized Stage 1's aux
#    propagator; the Stage-2-trained PropagatOR itself never had this
#    pressure applied directly, even though Stage 2 is exactly where the
#    final, deployed dynamics are fit.
# 3. --multistep and --k-pred-max DROPPED (back to Stage 1's plain,
#    fixed k_pred=2 -- Section 217's own original recipe). User's own
#    assessment: "rollout doesn't seem to help" -- 218's longer Stage-1
#    rollout (k ramped 2->8) didn't prevent divergence and arguably made
#    Stage-1-only chaos worse (faster onset than 217's own baseline), so
#    it's dropped rather than kept as a confound in this new comparison.
#
# --attn-window 9 KEPT from Section 219 (narrowed from 217/218's 18,
# ~3-site radius, matching this project's own correlation-length
# reasoning) -- "keep everything else the same" means the window stays
# narrowed, only the growth-control mechanism and the rollout curriculum
# change.
#
# Verified via real (--epochs 2) dry runs of BOTH stages before this
# launch: trains cleanly with --w-multistep-growth-barrier on both
# stages and plain k=2 Stage 1, no error. Full 774-test suite passes (8
# new tests for the log-barrier loss, including a direct hinge-vs-barrier
# gradient comparison confirming the hinge's zero-gradient-below-ceiling
# blind spot and the barrier's C1-continuous safeguard construction).
#
# Otherwise identical to Section 217/218/219: local_field encoder
# (n_sites=32, local_channels=3, d_latent=96) + masked_mlp propagator
# (markovian), Section 52's plain regularizers, L=100/NX=256 dt_snap=1.0.
#
# Stage 1 AND Stage 2 both run (the standing "always run Stage 2" rule).
# Lyapunov RuntimeError wrapped in try/except (Section 217's crash fix).
#
# NOT AUTO-LAUNCHED -- user asked to decide together before running this.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=section220_ks_localfield_maskedmlp_logbarrier_k2

echo "=== [1/4] Stage 1: local_field + masked_mlp propagator (markovian, attn_window=9), Section 52's regularizers PLUS --w-multistep-growth-barrier 0.1 (k=10, ceiling=75.0), plain k_pred=2 (no --multistep), L=100/NX=256 dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder local_field --aux-backbone masked_mlp --mode markovian \
  --attn-window 9 \
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
print('[$LABEL] (211: D3=0.1883/D_KY=23.04; 216: D3=0.2146/D_KY=22.59; 217: D3=0.2970/diverged; 218: D3=0.3661/diverged; 219: D3=0.4052/diverged)')
"
}

echo "=== [2/4] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== [3/4] Stage 2: warm-started from Stage 1, Section 52's exact schedule (k_max=12, 300 epochs) PLUS --w-multistep-growth-barrier (NEW for Stage 2 this time) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --w-multistep-growth-barrier 0.1 --multistep-growth-barrier-k 10 --multistep-growth-barrier-ceiling 75.0 \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [4/4] Stage 2 diagnostics ==="
_diagnose "$STAGE2_PROP" "stage2" > artifacts/logs/lyapunov_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/lyapunov_${STAGE2_TAG}.log

echo "=== Section 220 complete ==="
