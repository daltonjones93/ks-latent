#!/bin/zsh
# User-directed 2026-09-22: "yes build it" (Stage 2 continuation for
# Section 197 -- N=64 Lorenz-96, F=4.2, dt_snap=1.0, local_field encoder
# n_sites=8/channels=2 -> d_latent=16, dense MLP propagator markovian).
#
# Warm-starts from Section 197's own Stage-1 checkpoints (AE + aux
# propagator) and continues training the propagator alone (frozen
# encoder, per this project's established Stage-1/Stage-2 split) with a
# longer K-step curriculum -- Stage 1 never goes past k_pred_max=2 by
# default; Stage 2 is where the real multi-step rollout horizon lives.
#
# Curriculum: k_max=12, k_warmup_epochs=210, k_mid=8, k_mid_epochs=175,
# 300 epochs -- REUSED VERBATIM from this project's own established KS
# recipe (Sections 167/192/193's own Stage 2), not re-derived for L96.
# Deliberate choice, flagged honestly: at dt_snap=1.0 (~8% of this
# system's own Lyapunov time, tau~12.5), k_max=12 covers ~12 model-time-
# units (~96% of one Lyapunov time) -- a MUCH larger fraction of a
# Lyapunov time than KS's own k_max=12 covered there (dt_snap=0.25,
# tau~11.6 -> ~26% of a Lyapunov time). No strong reason yet to believe
# this mismatch matters one way or the other; reusing the validated
# curriculum as a first data point is cheaper than re-deriving a
# L96-specific one from scratch, and the gap is noted here for whoever
# reads this back.
#
# No pde_head (--pde-distill never used anywhere in this L96 line so
# far), no delta_cap, none of the newer spectrum-shape regularizers --
# same bare/baseline recipe as 197's own Stage 1, isolating "does more
# training + a longer horizon curriculum improve on 197's own
# lambda1=underestimate" as the only question this run answers.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section197_lorenz96_n64_f4p2_dtsnap1
DATASET=artifacts/datasets/lorenz96_trajectories_n64_f4.2_dtsnap1.h5
AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [1/2] Stage 2: Lorenz-96 N=64 F=4.2 dt_snap=1.0, warm-started from Section 197's own Stage 1, dense MLP propagator, k_max=12/k_warmup=210/k_mid=8/k_mid_epochs=175, 300 epochs ==="
STAGE2_TAG="${TAG}_stage2_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --dataset "$DATASET" \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [2/2] Diagnostics: reconstruction quality, standalone + multi-IC rollout (k=200, ~16 Lyapunov times), real Lyapunov spectrum vs. true L96 N=64/F=4.2 reference (lambda1=0.080, n_positive=5/64, D_KY=11.3) ==="
mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$STAGE2_PROP')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('d_latent:', d_latent)

DATASET = '$DATASET'
with h5py.File(DATASET,'r') as f:
    traj20 = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, N = traj20.shape
with torch.no_grad():
    z_all = ae.encode(traj20.reshape(n*T, N)).reshape(n, T, -1)
z0b = z_all[:,0,:]

with torch.no_grad():
    traj_roll = prop.rollout(z0b, z0b, k=200)
finite = torch.isfinite(traj_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('multi-IC (20) rollout first non-finite step:', first_bad)
maxz_per_t = traj_roll.abs().amax(dim=(0,2))
for t in [0,5,10,30,60,100,150,199]:
    if t < traj_roll.shape[1]:
        print(' t=%d  max|z| across all 20 ICs: %.4f' % (t, maxz_per_t[t].item()))
final_states = traj_roll[:,-1,:]
pairwise = torch.cdist(final_states, final_states)
print('final-state pairwise distance (min excl. diag, max, mean):',
      (pairwise + torch.eye(20)*1e9).min().item(), pairwise.max().item(), pairwise.mean().item())

with h5py.File(DATASET,'r') as f:
    traj = torch.tensor(f['trajectories'][3], dtype=torch.float32)
with torch.no_grad():
    z01 = ae.encode(traj[:2]).numpy()
res = lyapunov_spectrum_latent_propagator(
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=200, qr_every=5,
    dt_snap=1.0, warmup_steps=20, seed=0, max_abs_state=1e3,
)
print('lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('D_KY: could not bracket --', e)
print('(true L96 N=64/F=4.2 reference: lambda1=0.080, n_positive=5/64, D_KY=11.3)')
print('(Section 197 Stage 1 comparison: see artifacts/logs/lyapunov_section197_lorenz96_n64_f4p2_dtsnap1.log)')
" > artifacts/logs/lyapunov_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/lyapunov_${STAGE2_TAG}.log

echo "=== Section 197 Stage 2 complete ==="
