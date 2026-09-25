#!/bin/zsh
# User-directed 2026-09-25: "I think we have the infrastructure to
# train a pde jointly alongside a propagator. can we please rerun all
# the settings of 224 as 225 and include this pde. Try to make sure
# this pde does not have explosive dynamics in the long term. and that
# it might exhibit chaos (regularize using the appropriate
# regularizers). Use a chebyshev basis too, I think that may be more
# stable. try this, we can see how it looks."
#
# Exact Section 224 recipe (local_field n_sites=16/local_channels=3,
# local_mlp propagator tokens aligned to sites, attn_window=3, Section
# 52's plain regularizers, plain k_pred=2, --full-propagator, --amp),
# PLUS `--pde-distill`: a SEPARATE, distilled `pde_head`
# (backbone="spectral_pde_raw", the only existing "train a PDE jointly
# with a propagator" infrastructure in this codebase -- researched
# thoroughly before this launch, see docs/RESULTS.md's writeup) trained
# alongside it.
#
# Stability/chaos design, reasoned from this project's own documented
# history of every prior pde_head attempt (Sections 138-193, none of
# which combined it with local_mlp or with the modern 2000-step
# standalone-rollout validation standard):
#
# - `--aux-backbone spectral_pde_raw` (pde_head AS the primary
#   propagator) has a clean, decisive NEGATIVE precedent (Sections
#   169/171: collapsed to D_KY=0, DA skill=0.82, worse than free-
#   running) -- NOT used here. `--pde-distill` (a SEPARATE, distilled
#   head next to a free aux) is the only version with any positive
#   precedent (Sections 138-140).
# - Stage 1: kept DETACHED (no --pde-mutual) -- `pde_distill_detach_
#   target=True` is the default, protecting local_mlp's own gradient
#   from pde_head's "pressure toward simplicity" (the exact mechanism
#   blamed for this project's own H-PROP fixed-point-collapse finding).
#   Stage 1's local_mlp/encoder training is otherwise IDENTICAL to
#   Section 224's own recipe.
# - Stage 2: Stage2TrainingConfig's own w_pde_distill mechanism is
#   UNCONDITIONALLY mutual (confirmed directly in ks_latent/training/
#   loops.py's train_stage2 docstring -- there is no detached option in
#   Stage 2, a deliberate design: "we're really just training the
#   propagator right, the pde_head training is acting as a
#   regularization term", per earlier user direction). --freeze-
#   propagator (pde_head-only, no propagator refinement at all) was
#   REJECTED here specifically because it would skip Section 224's own
#   Stage-2 propagator refinement entirely, breaking the "rerun 224's
#   settings" requirement -- so Stage 2 here IS mutual, a real,
#   acknowledged risk, mitigated by the regularizers below.
# - --pde-field-kind chebyshev (user's explicit request), --pde-poly-
#   degree 2 (matches KS's own quadratic nonlinearity and Section 140's
#   own precedent), --pde-integrator euler (etdrk4 empirically diverged
#   for spectral_pde_raw, confirmed in propagator.py's own docstring --
#   never use etdrk4 here).
# - Explosive-dynamics guards: --pde-poly-no-constant (prevents drift-
#   into-a-constant, the Section 165 failure mode) and --pde-poly-
#   exclude-nonconservative (architecturally exact anti-drift, matches
#   true KS's own exact ∫u dx conservation) -- BOTH confirmed valid for
#   field_kind=chebyshev (their own CLI help text says "polynomial/
#   chebyshev", unlike --pde-poly-stable-leading, whose help text says
#   "polynomial only" specifically -- SKIPPED here since its sign-
#   derivation is stated for monomial coefficients and was never
#   verified for Chebyshev-space ones; the user's own "chebyshev is
#   more stable" intuition is only partially borne out by the code
#   (real conditioning benefit is basis-agnostic-fit-quality, not an
#   architectural boundedness guarantee on its own -- see docs/RESULTS.md).
# - Chaos-preservation: --w-pde-spectrum-shape (pde_head's own known
#   failure mode, Section 192's own motivation: "genuinely chaotic...
#   but pde_head... DECAYS toward a fixed point... over 200 steps" even
#   when the real propagator stays healthy) + --w-pde-energy-floor
#   (Section 167's complementary anti-collapse floor on pde_head's own
#   rollout amplitude). Both basis-agnostic (operate on Jacobian
#   singular values / rollout statistics, not on any specific
#   coefficient), applied in BOTH stages.
#
# Verified via real (--epochs 2) dry runs of both stages before this
# launch: trains cleanly end to end. One real, informative warning
# already visible even at 2 epochs: pde_head's own unsupervised rollout
# goes non-finite at k=30 by epoch 1 -- handled gracefully (skips that
# batch's energy-floor term, doesn't crash), but a live signal this
# pde_head is genuinely prone to blowing up early; whether the
# regularizers tame this over the full 200/300-epoch schedule is
# exactly what this run is testing.
#
# w_pde_distill=0.2 (between Section 140's 0.3 and a more conservative
# choice, given this is simultaneously a first-ever local_mlp
# combination AND a first-ever chebyshev-with-these-guards combination
# AND Stage 2 is unconditionally mutual -- three compounding sources of
# risk relative to any single prior attempt).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
TAG=section225_ks_localfield_localmlp_pdedistill_chebyshev

echo "=== [1/4] Stage 1: Section 224's exact recipe + --pde-distill (DETACHED, chebyshev, degree 2, euler, no-constant + exclude-nonconservative + spectrum-shape + energy-floor), --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder local_field --local-field-n-sites 16 --local-field-channels 3 \
  --aux-backbone local_mlp --mode markovian --aux-n-tokens 16 --attn-window 3 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --pde-distill --w-pde-distill 0.2 \
  --pde-field-kind chebyshev --pde-poly-degree 2 --pde-integrator euler \
  --pde-poly-no-constant --pde-poly-exclude-nonconservative \
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
print('[$LABEL] (true L100 target: D_KY in [21,24]; 224 own: D_KY=22.14, D3=0.4313, bounded)')

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

echo "=== Section 225 complete ==="
