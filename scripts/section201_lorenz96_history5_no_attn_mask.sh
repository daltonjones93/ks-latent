#!/bin/zsh
# User-directed 2026-09-23: "kill the current training and run 199 but
# with like 5 steps of history in the propagator. use dt_snap 0.1. also
# don't use any attention masking in the ViT."
#
# Context: the "Inertial Manifold Divide" diagnostic (same day) measured
# the TRUE physical Lyapunov spectra of KS and Lorenz-96 at matched state
# dimension (N=64) and found KS has a sharp spectral gap (sum of all 64
# exponents = -4180, dominated by a strongly-damped high-k tail) while
# L96 does not (sum = exactly -64, damping spread almost uniformly across
# all 64 directions -- no analogue of KS's k^2-k^4 scale-selective
# dissipation, hence no finite-dimensional inertial manifold). That
# predicts L96 genuinely needs propagator MEMORY (Mori-Zwanzig) that a
# Markovian latent map structurally cannot supply -- weakly supported
# already by Section 198 (mode=two_step, 1 step of history, D_KY=2.01)
# beating Section 199 (mode=markovian, 0 history, D_KY=1.35) at identical
# d_latent=20. This section pushes that lever hard: --mode history
# --n-history 6 (5 PAST states + the current one = "5 steps of history",
# this codebase's own n_history convention is "total states including
# current" -- see PropagatorConfig's docstring). Requires --aux-backbone
# vit (train_stage1_patched.py's own --mode help: "'history' ... requires
# --aux-backbone vit").
#
# "don't use any attention masking in the ViT": --attn-window's own
# default is already None = full (unmasked) attention -- Section 199 was
# the one that turned windowing ON (--attn-window 4 --token-window 16,
# copied verbatim from KS's own Section 52 recipe). Both flags are simply
# OMITTED here, which restores full attention for the encoder AND (since
# --prop-attn-window falls back to --attn-window when not given directly,
# per its own CLI help) the mode=history aux propagator's own ViT-style
# attention over its n_history*n_tokens tokens. --token-window (overlapping
# patch-embedding input width) is a separate knob from attention masking
# and is likewise left at its own default (None = non-overlapping) rather
# than carried over from 199 -- nothing about it was specifically
# requested, and it was tuned for KS's 32-token case, not this one's 8.
#
# Everything else copied from Section 199 unchanged: d_latent=20 (same
# ratio vs true D_KY=11.3 for direct comparability), w_var=0.02,
# w_spatial=0.01 signed, w_logdet=0.0035, NO delta_cap, pos_encoding=linear
# (still applies to --aux-backbone vit, orthogonal to attention masking),
# same Lorenz-96 N=64/F=4.2/dt_snap=0.1 dataset ("use dt_snap 0.1" is
# already what this dataset is -- the dt_snap=1.0 variant used by Section
# 197 is NOT used here), same 40/60-epoch Stage-1/Stage-2 schedule.
#
# _diagnose() below is adapted for mode="history": the rollout uses
# prop.rollout_history(z_hist, k) (z_hist: (B, n_history, d), oldest to
# newest) instead of prop.rollout(z_prev, z_curr, k), and the Lyapunov
# call uses lyapunov_spectrum_latent_propagator(..., mode='history', ...)
# with a genuine n_history-length encoded window as initial_pair, per
# that function's own docstring -- 'single_state'/'two_step' are NOT
# valid tangent-map approximations for a history propagator.
#
# Section 200 (plain MLP encoder + minimal regularization, launched
# earlier the same day) had already finished Stage 1 and self-terminated
# on its own diagnostics step (RuntimeError: reference trajectory escaped
# bound, max|z|~1000+) before this section was requested -- nothing to
# kill; noted here so the run isn't mistaken for silently abandoned. That
# result is itself informative (no variance/spatial regularization at all
# -> the rollout diverges rather than settling onto ANY bounded orbit,
# chaotic or not) and is separate from this section's own question.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section201_lorenz96_history5_no_attn_mask
DATASET=artifacts/datasets/lorenz96_trajectories_n64_f4.2.h5
N_HISTORY=6

echo "=== [1/3] Stage 1: ViT encoder (pos_encoding=linear, FULL attention -- no --attn-window), d_latent=20 + mode=history (n_history=6, 5 past + current) ViT-backbone propagator, w_var=0.02, w_spatial=0.01 signed, w_logdet=0.0035, NO delta_cap, on Lorenz-96 N=64 F=4.2 dt_snap=0.1, --amp, 40 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --nx 64 --d-latent 20 \
  --encoder vit --aux-backbone vit --mode history --n-history $N_HISTORY \
  --pos-encoding linear \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --full-propagator --amp \
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
print('[$LABEL] (Section 196 comparison, local_field d16 markovian: D_KY=4.93)')
print('[$LABEL] (Section 198 comparison, ViT d20 two_step (1 step history): D_KY=2.01)')
print('[$LABEL] (Section 199 comparison, ViT d20 markovian (0 history): D_KY=1.35, Stage2 collapsed to 0.00)')
"
}

echo "=== [2/3] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== [3/3] Stage 2: warm-started from Stage 1, NO delta_cap, --amp, k_max=12, 60 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_60ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --dataset "$DATASET" \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 60 --k-max 12 --k-warmup-epochs 42 --k-mid 8 --k-mid-epochs 35 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== Stage 2 diagnostics ==="
_diagnose "$STAGE2_PROP" "stage2" > artifacts/logs/lyapunov_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/lyapunov_${STAGE2_TAG}.log

echo "=== Section 201 complete ==="
