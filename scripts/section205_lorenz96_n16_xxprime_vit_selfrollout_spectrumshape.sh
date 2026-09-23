#!/bin/zsh
# RESULT (2026-09-23): Stage 1 itself collapsed -- lambda1=-4.51e-05,
# n_positive=0/8, D_KY=0.00 -- despite healthy reconstruction
# (val_recon_final=0.0836, smooth monotonic descent, no collapse
# plateau). User-directed "don't run stage 2 then" once this came in:
# Stage 2 (where --w-spectrum-shape-self actually applies) was never
# run -- warm-starting it from an already-collapsed propagator can't
# test whether that regularizer prevents collapse. Points at the ViT
# backbone itself (not the regularizer stack or the x+x' augmentation)
# as the likely cause -- see docs/RESULTS.md's "Sections 204/205" entry
# and docs/OPEN_QUESTIONS.md for the full writeup and the recommended
# next step (finish Section 204's plain-MLP variant as the actual
# controlled test).
#
# User-directed 2026-09-23: "kill 204, and replace the encoder and
# decoder with the ViT. No attention mask needed please."
#
# Exact copy of Section 204 (killed before completion -- see that
# script's header for the full context: Section 203's rerun confirmed
# --multistep as the Stage-1 collapse cause but Stage 2 still collapsed
# to D_KY=0.00 despite w_varmatch+w_spatial+w_logdet_rollout_latent; this
# line tests the one remaining candidate, --w-spectrum-shape-self, the
# self-rollout analogue of Section 193's pde_head-only mechanism, now
# generalized to the MAIN propagator and to mode="history") with ONLY
# the encoder/decoder and aux-propagator backbone changed:
#   --encoder mlp --aux-backbone mlp   ->   --encoder vit --aux-backbone vit
# "No attention mask needed": --attn-window/--prop-attn-window both
# already default to None (full attention) when omitted -- same as
# Section 201's own "don't use any attention masking in the ViT"
# request, satisfied there (and here) by simply never passing those
# flags, not by an explicit "off" value.
#
# --pos-encoding linear kept (not the vit default of "circular") --
# carries forward Sections 199/201/203's own established choice for this
# L96 line (user decision, KS side: "most real-world PDEs of interest are
# not periodic" -- ported here for cross-section comparability, even
# though L96 IS exactly periodic and 'circular' would be the more
# physically apt choice; not revisited here since the user's request was
# specifically about the encoder/decoder body and attention masking, not
# positional encoding).
#
# Sizing check: NX=32 (x+x', unchanged), --patch-size defaults to 8 ->
# encoder n_tokens=4 (32/8, divides evenly). --aux-n-tokens defaults to 4
# -> aux propagator tokenizes d_latent=8 into 4 tokens of width 2
# (divides evenly). Both defaults used unmodified, no override needed at
# this size.
#
# Verified via a real (--epochs 2) dry run of BOTH stages before this
# launch (vit+history Stage 1, no attn-window, pos_encoding=linear: recon
# 0.577->0.246 over 2 epochs, no error; Stage 2 with
# --w-spectrum-shape-self on: 2 epochs, no error) -- confirms the
# combination trains cleanly and that the ViT tokenization sizing above
# is actually consistent at d_latent=8/NX=32, not just arithmetically
# plausible.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section205_lorenz96_n16_xxprime_vit_selfrollout_spectrumshape
DATASET=artifacts/datasets/lorenz96_trajectories_n16_f6.0_xxprime.h5
N_HISTORY=6

echo "=== [1/3] Stage 1: ViT encoder/decoder (NX=32, x+x', pos_encoding=linear, FULL attention -- no --attn-window), d_latent=8, mode=history (n_history=6) ViT-backbone propagator, w_var=0.02, w_spatial=0.01 signed, w_logdet=0.0035, on Lorenz-96 N=16 F=6.0 dt_snap=0.1, --amp, 40 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --nx 32 --d-latent 8 \
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
print('[$LABEL] (true L96 N=16/F=6.0 reference (in x alone, 16-dim): lambda1=0.980, n_positive=5/16, D_KY=9.39 -- NOTE: d_latent=8 caps the achievable D_KY at 8)')
"
}

echo "=== [2/3] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== [3/3] Stage 2: warm-started from Stage 1, w_varmatch + w_spatial + w_logdet_rollout_latent (Section 203's own failed stack) PLUS w_spectrum_shape_self (the new self-rollout term, this section's actual test), --amp, k_max=12, 60 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_60ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --dataset "$DATASET" \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --w-varmatch 0.02 --w-varmatch-adaptive \
  --w-spatial 0.01 --spatial-signed \
  --w-logdet-rollout-latent 0.0035 \
  --w-spectrum-shape-self 0.4 --spectrum-shape-self-rollout-k 20 --spectrum-shape-self-n-samples 16 \
  --spectrum-shape-n-expand 3 --spectrum-shape-expand-target 1.1 --spectrum-shape-contract-floor 0.6 --spectrum-shape-two-sided \
  --amp \
  --epochs 60 --k-max 12 --k-warmup-epochs 42 --k-mid 8 --k-mid-epochs 35 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== Stage 2 diagnostics ==="
_diagnose "$STAGE2_PROP" "stage2" > artifacts/logs/lyapunov_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/lyapunov_${STAGE2_TAG}.log

echo "=== Section 205 complete ==="
