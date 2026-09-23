#!/bin/zsh
# User-directed 2026-09-23: "can we copy the model and settings from
# section 52 and try again on the lorenz 96 system." Section 52 (KS,
# 2026-09-01/02): ViT encoder (pos_encoding=linear, attn_window=4,
# token_window=16, patch_size=8 default -> d_latent=44 default), dense
# MLP propagator (markovian), w_var=0.02, w_spatial=0.01 (SIGNED),
# w_decorr=0, w_var_floor=0, w_logdet=0.0035, NO delta_cap, Stage 1 200
# epochs, Stage 2 warm-started k_max=12/300 epochs. Copied VERBATIM here
# -- same weights, same architecture flags, same schedule -- only the
# dataset changes (Lorenz-96 N=64, F=4.2, dt_snap=0.1 -- the SAME dataset
# Section 196 used, our best intrinsic-chaos result in this L96 line so
# far, for a direct apples-to-apples comparison of "does a different,
# previously-validated KS recipe do better here than 196's own local_field
# encoder did").
#
# At N=64, patch_size=8 (unchanged default) gives only 8 tokens (vs KS's
# own 32 at NX=256) -- attn_window=4/token_window=16 (both tuned for the
# 32-token case) are reused UNCHANGED, not rescaled for the much smaller
# token count; verified directly via a real dry run that this raises no
# error (token_window=16 > 8 tokens is evidently handled safely, not
# rejected).
#
# User-directed follow-up, before the first launch finished: "kill 198
# and restart with latent dim = 20, 44 is too large." d_latent=44 (this
# script's original, Section 52's own unmodified default) is a much more
# generous oversizing (ratio ~3.9x over L96 N=64/F=4.2's own true
# D_KY~11.3) than any other pairing tried in this L96 line -- 196's own
# local_field d_latent=16 (ratio ~1.4x) was already enough to produce
# genuine bounded chaos. --d-latent 20 (ratio ~1.77x vs true D_KY=11.3)
# tests a middle ground: still comfortably oversized, closer to 196's own
# working ratio than to 44's very generous one.
#
# SECOND follow-up: "do d_latent = 20 and make give the propagator
# history, increase by one step back" -- --mode two_step (this
# codebase's built-in "(z_{n-1}, z_n) -> z_{n+1}" option, exactly one
# step of history) REPLACES markovian; --aux-backbone dropped entirely
# (mode="two_step" always uses a plain MLPDeltaBody(2*d, d, ...)
# regardless of --aux-backbone, per LatentPropagator.__init__ -- passing
# --aux-backbone mlp alongside it would be silently ignored, so it's
# removed here rather than left in misleadingly). Verified directly via a
# real dry run + checkpoint inspection: prop.mode == 'two_step' as
# expected, d_latent == 20, no NaN over 3 epochs.
#
# THIRD follow-up: "why is 198 so much slower to train than 52? shouldn't
# it be faster?" -- diagnosed directly: the L96 F=4.2 dataset has 8x more
# snapshots per trajectory than KS's own (2001 vs 251, same 60
# trajectories -- a leftover consequence of sizing trajectory_time=200 at
# dt_snap=0.1 for F=4.2's slower dynamics, unrelated to today's
# architecture choice). Batches/epoch scales directly with total
# snapshots (`index_shuffle_batches(window_index.shape[0], ...)`), so
# each epoch here does ~8x more gradient updates than a KS epoch at the
# same epoch count -- confirmed via measured per-epoch wall time (52:
# 6.7s/epoch; 198's first attempt: 25.0s/epoch, "only" ~3.7x slower than
# 8x since the smaller NX=64/d_latent=20/8-token-ViT genuinely is cheaper
# per batch). User: "ah, I see then we can do 5x fewer epochs than the KS
# training. kill the current training and restart with this." Stage 1
# 200->40 epochs, Stage 2 300->60 epochs (k_warmup_epochs 210->42,
# k_mid_epochs 175->35, proportionally -- k_max/k_mid themselves are
# rollout-HORIZON values, not epoch counts, left unchanged at 12/8),
# checkpoint-every 20->4.
#
# Correctness fix threaded through _diagnose() below: the Lyapunov
# spectrum measurement now uses mode='two_step' (the library's EXACT
# tangent map on the true 2*d_latent-dimensional (z_{n-1},z_n) phase
# space), not 'single_state' (described in ks_latent.analysis.lyapunov's
# own docstring as "the z->z approximation" -- correct only for a
# genuinely Markovian propagator, which this one no longer is). z01 (two
# real consecutive encoded states) already has exactly the shape this
# mode needs; n_directions stays at d_latent (not 2*d_latent) as a
# reasonable subset, consistent with every other section's own choice.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section198_lorenz96_section52_recipe
DATASET=artifacts/datasets/lorenz96_trajectories_n64_f4.2.h5

echo "=== [1/3] Stage 1: Section 52's recipe (ViT encoder, pos_encoding=linear, attn_window=4, token_window=16, d_latent=20) + mode=two_step (one step of history) propagator, w_var=0.02, w_spatial=0.01 signed, w_logdet=0.0035, NO delta_cap, on Lorenz-96 N=64 F=4.2 dt_snap=0.1, --amp, 40 epochs (1/5 of KS's own 200 -- this dataset has 8x more snapshots/trajectory, so 40 epochs here covers more gradient updates than 200 would on KS) ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --nx 64 --d-latent 20 \
  --encoder vit --mode two_step \
  --pos-encoding linear --attn-window 4 --token-window 16 \
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
    prop, z01, mode='two_step', n_directions=d_latent, n_steps=2000, qr_every=10,
    dt_snap=0.1, warmup_steps=200, seed=0, max_abs_state=1e3,
)
print('[$LABEL] lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('[$LABEL] D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('[$LABEL] D_KY: could not bracket --', e)
print('[$LABEL] (true L96 N=64/F=4.2 reference: lambda1=0.080, n_positive=5/64, D_KY=11.3)')
print('[$LABEL] (Section 196 comparison, local_field d_latent=16: lambda1=0.056, n_positive=3/16, D_KY=4.93)')
"
}

echo "=== [2/3] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== [3/3] Stage 2: warm-started from Stage 1, NO delta_cap, --amp, k_max=12, 60 epochs (1/5 of KS's own 300, same reasoning) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
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

echo "=== [bonus] Hovmoller visualization (Stage 1 and Stage 2) ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$AUX" --dataset "$DATASET" \
  --dt-snap 0.1 --L 64 --rollout-steps 300 --tag "${TAG}_stage1" \
  > artifacts/logs/visualize_${TAG}_stage1.log 2>&1
cat artifacts/logs/visualize_${TAG}_stage1.log
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$STAGE2_PROP" --dataset "$DATASET" \
  --dt-snap 0.1 --L 64 --rollout-steps 300 --tag "${TAG}_stage2" \
  > artifacts/logs/visualize_${TAG}_stage2.log 2>&1
cat artifacts/logs/visualize_${TAG}_stage2.log

echo "=== Section 198 complete ==="
