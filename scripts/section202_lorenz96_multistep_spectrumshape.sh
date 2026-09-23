#!/bin/zsh
# User-directed 2026-09-23: "can we combine ideas 1 and 2 and launch
# another training?" -- combining, from the Section 201 Stage-2-collapse
# discussion:
#
# Idea 1: `--w-spectrum-shape` (Stage1TrainingConfig/Stage2TrainingConfig,
# Sections 131/132) -- shapes the propagator's own per-step Jacobian
# singular-value spectrum directly (top n_expand toward an expansive
# target, the rest toward a lower contraction floor), added originally
# for the IDENTICAL KS-side symptom ("so it looks like it collapsed in
# stage 2? please wire that fix you mentioned into stage 2"). Retargeted
# for L96 here: `--spectrum-shape-n-expand 5` (this system's own true
# n_positive=5/64 at F=4.2, replacing KS's n_expand=11/13 convention),
# `--spectrum-shape-expand-target 1.1 --spectrum-shape-contract-floor 0.6
# --spectrum-shape-two-sided` (the LATEST/most-converged KS values,
# Sections 186-191, not the original 131/132 numbers -- two_sided fixes a
# real bug the one-sided floor had: once singular values sit ABOVE the
# target, `relu(target - sv)` is identically zero and provides no
# corrective pressure at all). These KS-tuned numeric values (weight 0.4,
# target 1.1, floor 0.6) are NOT re-derived from L96's own measured
# spectrum -- reused as a first approximation, flagged as unverified for
# this system, revisit if this run's results look off.
#
# KNOWN, STATED LIMITATION (see docs/OPEN_QUESTIONS.md): `--w-spectrum-
# shape` currently only supports `mode="markovian"` --
# `_propagator_step_jacobian_singular_values` operates on `step_one`.
# Section 201's best result (D_KY=11.94) used `mode="history"`
# (n_history=6) -- NOT combined here. This run tests whether idea 1 + 2
# fix the collapse on a markovian propagator FIRST (cheap, zero new code,
# reuses proven-on-KS machinery); extending the regularizer to
# `mode="history"` (computing the Jacobian of `step_history` w.r.t. only
# the newest state in the window) is real, not-yet-done work, worth doing
# ONLY if this run shows the mechanism transfers to L96 at all.
#
# Idea 2: `--multistep --full-propagator` -- real-sized aux propagator
# (hidden=128, n_blocks=3, same sizing either flag gives), rollout ramped
# k_pred 2->8 over the first 30% of epochs, instead of the brief's tiny
# fixed-k_pred=2 auxiliary propagator. Mechanistic rationale (not just
# "more training"): Stage 1 trains the propagator JOINTLY with
# reconstruction + the geometry regularizers (w_var/w_spatial/w_logdet),
# a materially more constrained optimization landscape than Stage 2's
# propagator-only fine-tune -- pushing more of the multi-step dynamics
# fitting into Stage 1's regime, rather than handing it entirely to Stage
# 2, is the actual hypothesis being tested. (Caveat stated to the user:
# extending Stage-1 rollout was found INERT for reconstruction accuracy
# on KS -- a different outcome variable than chaos preservation, which
# was never tested there; no direct precedent either way for this use.)
#
# Otherwise: Section 201's validated config minus mode=history -- ViT
# encoder, pos_encoding=linear, FULL/unmasked attention (no --attn-window/
# --token-window, per Section 201's own finding), d_latent=20, aux-
# backbone=mlp (markovian's own default), w_var=0.02, w_spatial=0.01
# signed, w_logdet=0.0035, NO delta_cap, same Lorenz-96 N=64/F=4.2/
# dt_snap=0.1 dataset, same 40/60-epoch Stage-1/Stage-2 schedule.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section202_lorenz96_multistep_spectrumshape
DATASET=artifacts/datasets/lorenz96_trajectories_n64_f4.2.h5
SPECTRUM_ARGS=(--w-spectrum-shape 0.4 --spectrum-shape-n-expand 5 --spectrum-shape-expand-target 1.1 --spectrum-shape-contract-floor 0.6 --spectrum-shape-n-samples 32 --spectrum-shape-two-sided)

echo "=== [1/3] Stage 1: ViT encoder (pos_encoding=linear, full attention), d_latent=20, mode=markovian, --multistep (k_pred 2->8) + --w-spectrum-shape (n_expand=5, target=1.1, floor=0.6, two_sided), w_var=0.02, w_spatial=0.01 signed, w_logdet=0.0035, on Lorenz-96 N=64 F=4.2 dt_snap=0.1, --amp, 40 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --nx 64 --d-latent 20 \
  --encoder vit --aux-backbone mlp --mode markovian \
  --pos-encoding linear \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --multistep --full-propagator "${SPECTRUM_ARGS[@]}" --amp \
  --epochs 40 --checkpoint-every 4 \
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

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$PROP_PATH')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('[$LABEL] d_latent:', d_latent)

DATASET = '$DATASET'
with h5py.File(DATASET,'r') as f:
    traj20 = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, N = traj20.shape
with torch.no_grad():
    z_all = ae.encode(traj20.reshape(n*T, N)).reshape(n, T, -1)
z0b = z_all[:,0,:]

with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=2000)
finite = torch.isfinite(traj_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('[$LABEL] multi-IC (20) rollout first non-finite step:', first_bad)
maxz_per_t = traj_roll.abs().amax(dim=(0,2))
for t in [0,50,100,300,600,1000,1500,1999]:
    if t < traj_roll.shape[1]:
        print('[$LABEL] t=%d  max|z| across all 20 ICs: %.4f' % (t, maxz_per_t[t].item()))
final_states = traj_roll[:,-1,:]
pairwise = torch.cdist(final_states, final_states)
print('[$LABEL] final-state pairwise distance (min excl. diag, max, mean):',
      (pairwise + torch.eye(20)*1e9).min().item(), pairwise.max().item(), pairwise.mean().item())

with h5py.File(DATASET,'r') as f:
    traj = torch.tensor(f['trajectories'][3], dtype=torch.float32)
with torch.no_grad():
    z01 = ae.encode(traj[:2]).numpy()
res = lyapunov_spectrum_latent_propagator(
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=2000, qr_every=10,
    dt_snap=0.1, warmup_steps=200, seed=0, max_abs_state=1e3,
)
print('[$LABEL] lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('[$LABEL] D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('[$LABEL] D_KY: could not bracket --', e)
print('[$LABEL] (true L96 N=64/F=4.2 reference: lambda1=0.080, n_positive=5/64, D_KY=11.3)')
print('[$LABEL] (Section 199 comparison, ViT d20 markovian, no spectrum-shape: stage1 D_KY=1.35, stage2 D_KY=0.00)')
print('[$LABEL] (Section 201 comparison, ViT d20 history n=6, no spectrum-shape: stage1 D_KY=11.94, stage2 D_KY=0.00)')
"
}

echo "=== [2/3] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== [3/3] Stage 2: warm-started from Stage 1, --w-spectrum-shape (same L96-adapted values) carried into Stage 2, NO delta_cap, --amp, k_max=12, 60 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_60ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --dataset "$DATASET" \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  "${SPECTRUM_ARGS[@]}" \
  --amp \
  --epochs 60 --k-max 12 --k-warmup-epochs 42 --k-mid 8 --k-mid-epochs 35 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== Stage 2 diagnostics ==="
_diagnose "$STAGE2_PROP" "stage2" > artifacts/logs/lyapunov_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/lyapunov_${STAGE2_TAG}.log

echo "=== Section 202 complete ==="
