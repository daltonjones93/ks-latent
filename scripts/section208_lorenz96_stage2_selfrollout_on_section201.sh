#!/bin/zsh
# RESULT (2026-09-23): still collapsed -- lambda1=-0.0028, n_positive=
# 0/20, D_KY=0.0 -- despite warm-starting from the best Stage-1
# checkpoint this L96 line has produced (D_KY=11.94) and despite
# w_spectrum_shape_self being active alongside the already-tried stack.
# Notable nuance: settled onto a BOUNDED, non-trivial orbit (max|z|
# stays ~2.5-3.0, final-state pairwise spread stays real, 0.82-10.37)
# rather than a literal fixed point -- a stable limit cycle, not chaos,
# but zero positive Lyapunov exponents either way. Fifth independent
# confirmation of Stage-2 collapse (197, 199, 201, 203, 208); closes
# out the regularizer-toolkit line of inquiry -- see docs/RESULTS.md's
# "Section 208" entry and docs/OPEN_QUESTIONS.md for the full writeup
# and candidate directions OUTSIDE the existing regularizer toolkit
# (w_pred=0 "two_stage" experiment; a distributional rollout objective;
# direct Lyapunov-spectrum optimization as the primary Stage-2 loss).
#
# User-directed 2026-09-23: "let's try 1 then 2" -- option 1 of two
# proposed next steps after Section 207 found x' augmentation hurts
# propagator chaos (D_KY 11.94 -> 2.19) and was abandoned ("clearly we
# shouldn't use x'"). Option 1: test --w-spectrum-shape-self (the
# self-rollout main-propagator regularizer built in Section 204, still
# never validated against a genuinely chaotic Stage-1 checkpoint) by
# warm-starting Stage 2 DIRECTLY from Section 201's own Stage-1
# checkpoint -- no new Stage-1 run needed, it already exists on disk
# (D_KY=11.94, essentially matching the true L96 N=64/F=4.2 reference of
# 11.3) and was the best L96 Stage-1 result in this entire line.
#
# Same Stage-2 regularizer stack as Section 203's rerun (w_varmatch +
# w_spatial + w_logdet_rollout_latent, the combination that alone still
# collapsed Stage 2 to D_KY=0.00 on a WEAKER Stage-1 checkpoint,
# D_KY=2.117) PLUS w_spectrum_shape_self on top -- the actual, still-
# untested question this whole sub-thread (Sections 204-207) has been
# trying to answer. Propagator is mode=history (n_history=6),
# aux-backbone=vit, loaded from --init-prop-checkpoint (architecture
# flags ignored, governed by the checkpoint) -- _propagator_spectrum_
# shape_self_pool dispatches on propagator.mode, so "history" is
# supported directly, no special-casing needed here.
#
# spectrum-shape-n-expand=5 (not 3, unlike Sections 204-206's d_latent=8
# experiments): matches Section 203's own precedent AND Section 201's
# own achieved n_positive=4/20 almost exactly, at this run's actual
# d_latent=20.
#
# Dataset: back to the plain x-only N=64/F=4.2 dataset (NOT the x'
# interleaved one from Section 207) -- Section 201's own AE/propagator
# were trained on it, so Stage 2 must use the same one.
#
# Verified via a real (--epochs 2) dry run before this launch: trains
# without error.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section208_lorenz96_stage2_selfrollout_on_section201
DATASET=artifacts/datasets/lorenz96_trajectories_n64_f4.2.h5
AE=artifacts/stage1_ae_patched_full_section201_lorenz96_history5_no_attn_mask.pt
AUX=artifacts/stage1_prop_full_section201_lorenz96_history5_no_attn_mask.pt

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
n_history = prop.cfg.n_history
print('[$LABEL] d_latent:', d_latent, ' n_history:', n_history, ' mode:', prop.mode)

DATASET = '$DATASET'
with h5py.File(DATASET,'r') as f:
    traj20 = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, N = traj20.shape
with torch.no_grad():
    z_all = ae.encode(traj20.reshape(n*T, N)).reshape(n, T, -1)
z_hist0 = z_all[:, :n_history, :]  # (20, n_history, d), oldest to newest

with torch.no_grad():
    traj_roll = prop.rollout_history(z_hist0, k=2000)
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
    z_hist_single = ae.encode(traj[:n_history]).numpy()  # (n_history, d), oldest to newest
res = lyapunov_spectrum_latent_propagator(
    prop, z_hist_single, mode='history', n_directions=d_latent, n_steps=2000, qr_every=10,
    dt_snap=0.1, warmup_steps=200, seed=0, max_abs_state=1e3,
)
print('[$LABEL] lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('[$LABEL] D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('[$LABEL] D_KY: could not bracket --', e)
print('[$LABEL] (true L96 N=64/F=4.2 reference: lambda1=0.080, n_positive=5/64, D_KY=11.3)')
print('[$LABEL] (Section 201 comparison, stage1, same checkpoint used here: D_KY=11.94)')
print('[$LABEL] (Section 203 rerun comparison, stage2, d20 markovian, varmatch+spatial+logdet_rollout_latent, weaker D_KY=2.117 stage1: stage2 D_KY=0.00)')
"
}

echo "=== [1/2] Stage 2: warm-started DIRECTLY from Section 201's Stage-1 checkpoint (D_KY=11.94), w_varmatch + w_spatial + w_logdet_rollout_latent (Section 203's own failed stack) PLUS w_spectrum_shape_self (this section's actual test), --amp, k_max=12, 60 epochs ==="
mamba run -n da_env python scripts/train_stage2_patched.py \
  --dataset "$DATASET" \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --w-varmatch 0.02 --w-varmatch-adaptive \
  --w-spatial 0.01 --spatial-signed \
  --w-logdet-rollout-latent 0.0035 \
  --w-spectrum-shape-self 0.4 --spectrum-shape-self-rollout-k 20 --spectrum-shape-self-n-samples 16 \
  --spectrum-shape-n-expand 5 --spectrum-shape-expand-target 1.1 --spectrum-shape-contract-floor 0.6 --spectrum-shape-two-sided \
  --amp \
  --epochs 60 --k-max 12 --k-warmup-epochs 42 --k-mid 8 --k-mid-epochs 35 \
  --tag "$TAG" \
  > artifacts/logs/stage2_${TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${TAG}.pt

echo "=== [2/2] Stage 2 diagnostics ==="
_diagnose "$STAGE2_PROP" "stage2" > artifacts/logs/lyapunov_${TAG}.log 2>&1
cat artifacts/logs/lyapunov_${TAG}.log

echo "=== Section 208 complete ==="
