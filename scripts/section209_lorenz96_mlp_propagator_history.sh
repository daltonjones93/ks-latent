#!/bin/zsh
# User-directed 2026-09-23: "let's try 1 then 2" -- option 2 of two
# proposed next steps after Section 207 abandoned the x' augmentation.
# Rationale (given alongside the proposal): this project's own
# established KS-side finding (`ks_latent.models.propagator.
# _LocalMLPDeltaBody`'s docstring) is that plain dense-MLP propagators
# reliably beat self-attention-based ones -- softmax attention's
# convex-combination structure is inherently biased toward CONTRACTION,
# which fights the expansive directions chaos needs. Section 201 (ViT
# encoder AND ViT-backbone propagator, mode=history n_history=6) got
# genuinely good chaos (D_KY=11.94) at Stage 1, but its propagator is
# still self-attention-based -- untested whether an MLP propagator with
# the SAME history depth, on the SAME ViT-encoded latent, does as well
# or better, and (the more important question, per Section 205's own
# ViT-propagator collapse) whether it makes a more robust foundation for
# Stage 2 specifically.
#
# EXACT copy of Section 201's Stage 1 command with ONE variable changed:
# --aux-backbone vit -> --aux-backbone mlp (encoder stays --encoder vit,
# same d_latent=20, mode=history n_history=6, pos_encoding=linear, full
# attention -- no --attn-window, same w_var/w_spatial/w_logdet
# regularizers, same 40-epoch schedule). A clean single-variable
# ablation of the propagator backbone alone, holding the encoder/data/
# regularizers/schedule all fixed.
#
# Verified via a real (--epochs 2) dry run before this launch: recon
# 0.104 -> 0.0158 over 2 epochs (val_recon_final=0.0212) -- close to,
# though not quite matching, Section 201's own 2-epoch trajectory
# (0.110 -> ~0.004 by epoch 10) at this early a checkpoint; full 40
# epochs needed for a fair comparison.
#
# Stage 1 ONLY this launch (matching this arc's own "check chaos before
# Stage 2" discipline) -- if this recovers comparable or better D_KY
# than Section 201's 11.94, it becomes the preferred foundation for
# future Stage-2 regularizer tests (in place of Section 201's own
# ViT-propagator checkpoint used by Section 208), since an MLP
# propagator does not carry self-attention's documented collapse risk.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section209_lorenz96_mlp_propagator_history
DATASET=artifacts/datasets/lorenz96_trajectories_n64_f4.2.h5
N_HISTORY=6

echo "=== [1/2] Stage 1: ViT encoder (pos_encoding=linear, FULL attention -- no --attn-window), d_latent=20, mode=history (n_history=6) MLP-backbone propagator (Section 201's exact recipe, --aux-backbone vit -> mlp), w_var=0.02, w_spatial=0.01 signed, w_logdet=0.0035, on Lorenz-96 N=64 F=4.2 dt_snap=0.1, --amp, 40 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --nx 64 --d-latent 20 \
  --encoder vit --aux-backbone mlp --mode history --n-history $N_HISTORY \
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
print('[$LABEL] (Section 201 comparison, SAME recipe with --aux-backbone vit instead of mlp: stage1 D_KY=11.94)')
"
}

echo "=== [2/2] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== Section 209 Stage 1 complete (Stage 2 deferred pending this result) ==="
