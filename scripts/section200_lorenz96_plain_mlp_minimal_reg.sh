#!/bin/zsh
# User-directed 2026-09-23: "I guess maybe I want to go back to the
# drawing board then. Can we try training the L96 system using just
# simple MLPs for encoder decoder and propagator. moreover, remove any
# regularization aside from a small logdet term."
#
# Motivation: Section 199 (ViT encoder, Section 52's KS recipe, all of
# w_var/w_spatial/w_logdet on) reconstructed the field ~6x better than
# Section 196 (local_field encoder) but produced a WEAKER Stage-1
# propagator (D_KY=1.35 vs 196's 4.93) and, like every Stage-2 run tried
# on this system so far, collapsed to D_KY=0 in Stage 2. Rather than
# keep porting more of the KS line's own architecture/regularizer choices
# -- which were tuned for KS, not L96, and have not obviously helped here
# -- this section resets to the simplest possible architecture
# (`--encoder mlp`, the plain "Track A" MLP autoencoder; `--aux-backbone
# mlp` + `--mode markovian`, the plain dense-MLP propagator, both already
# this script's own defaults) and strips every optional regularizer this
# codebase has accumulated (w_var, w_decorr, w_spatial, w_var_floor, all
# explicitly 0; no delta_cap; no pde_head/spectrum-shape/lowpass/temporal-
# floor terms anywhere) down to just `--w-logdet 0.0035` -- the one term
# with a clear, simple rationale (push the WHOLE latent covariance
# eigenspectrum away from zero, preventing rank collapse) at the same
# small weight Section 52/198/199 already validated on KS.
#
# Same dataset as 196-199 (Lorenz-96 N=64, F=4.2, dt_snap=0.1 -- this
# system's best-characterized chaotic operating point in this L96 line)
# and the same d_latent=20 (ratio ~1.77x over the true D_KY=11.3
# reference, the same middle-ground sizing 198/199 used) for direct
# comparability against the existing comparison table. Same 40/60-epoch
# Stage-1/Stage-2 schedule (this dataset's own snapshot density, see
# Section 198's own diagnosis of why 40/60 replaces KS's usual 200/300).
#
# Includes the 2026-09-23 encode_dataset_with_shifts MPS nan fix (now
# permanent in ks_latent/training/loops.py) -- no special handling needed
# here, Stage 2 will simply pick it up.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section200_lorenz96_plain_mlp_minimal_reg
DATASET=artifacts/datasets/lorenz96_trajectories_n64_f4.2.h5

echo "=== [1/3] Stage 1: plain MLP encoder/decoder (Track A) + plain dense MLP propagator (markovian), NO regularization except w_logdet=0.0035, on Lorenz-96 N=64 F=4.2 dt_snap=0.1, --amp, 40 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --nx 64 --d-latent 20 \
  --encoder mlp --aux-backbone mlp --mode markovian \
  --w-decorr 0 --w-var 0 --w-spatial 0 --w-var-floor 0 --w-logdet 0.0035 \
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
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=2000, qr_every=10,
    dt_snap=0.1, warmup_steps=200, seed=0, max_abs_state=1e3,
)
print('[$LABEL] lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('[$LABEL] D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('[$LABEL] D_KY: could not bracket --', e)
print('[$LABEL] (true L96 N=64/F=4.2 reference: lambda1=0.080, n_positive=5/64, D_KY=11.3)')
print('[$LABEL] (Section 196 comparison, local_field d_latent=16: lambda1=0.056, n_positive=3/16, D_KY=4.93)')
print('[$LABEL] (Section 199 comparison, ViT d_latent=20: stage1 lambda1=0.0073/D_KY=1.35, stage2 D_KY=0.00)')
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

echo "=== Section 200 complete ==="
