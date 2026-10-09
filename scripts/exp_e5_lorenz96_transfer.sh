#!/bin/zsh
# E5 (docs/ks_local_field_ltransfer.tex): does the L-transfer mechanism
# generalize past KS, or is it a KS idiosyncrasy? Lorenz-96 is the
# cheapest second rung: it's periodic and 1D like KS (so the SAME
# `local_field`/`local_mlp` architecture and the SAME
# `load_autoencoder_checkpoint_resized`/`load_propagator_checkpoint_
# resized` machinery apply with zero code changes -- verified directly:
# neither function has any KS-PDE-specific logic, both operate purely on
# `KSAutoencoderLocalField`'s/`_LocalMLPDeltaBody`'s array-length-agnostic
# conv shapes), and it has a comparably clean extensivity law
# (D_KY grows with N at fixed forcing F, roughly linearly in the chaotic
# regime, measured directly by this project's own Benettin/QR code per
# an F-sweep in scripts/section196_lorenz96_n64_f4p2_densemlp.sh's own
# header: F=4.2 at N=64 gives D_KY=11.3).
#
# IMPORTANT CORRECTION to docs/PART_4_3_SUMMARY.md's own text (which
# claimed L96 "already has a validated local-field checkpoint" as a
# reason this rung would be cheap): checked directly against
# docs/RESULTS.md before writing this script -- that is NOT accurate.
# Every L96 local-field attempt found in the project's own history
# (Sections 195/196/198/199/201/203/204/205) either used a DENSE `mlp`
# propagator (not `local_mlp` -- NOT weight-shared, so NOT N-transferable
# even though the encoder was `local_field`) or hit the same repeated
# Stage-2 chaos-collapse failure mode documented at length in
# docs/RESULTS.md (four independent confirmations, no working fix found
# in that line of investigation). Section 196's own best local_field
# number undershoots its true reference by more than half (D_KY=4.93
# measured vs. D_KY=11.3 true, using a DENSE propagator besides). There
# is currently NO validated, N-transfer-capable L96 checkpoint anywhere
# in this project -- this script builds one from scratch, using KS
# Section 224's exact winning recipe (the one architecture combination
# that has actually worked, local_field + local_mlp) transplanted onto
# the L96 dataset, rather than assuming a shortcut that doesn't exist.
#
# Consequence: this is the highest-risk of the five E1-E5 experiments --
# it may simply hit the same Stage-2 collapse every prior L96 attempt in
# this project did, in which case the honest reportable outcome is "the
# L-transfer mechanism could not even be tested on a second system
# because no chaotic L96 checkpoint could be trained," not a transfer
# failure per se. Run this LAST, after E1-E4, and budget real
# troubleshooting time if Stage 2 collapses again.
#
# Usage: ./scripts/exp_e5_lorenz96_transfer.sh
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/lorenz96_trajectories_n64_f4.2.h5
TAG=exp_e5_lorenz96_n64_f4.2_localfield_localmlp
TRUE_DKY_N64=11.3  # docs/RESULTS.md, F=4.2 reference (Section 196's own header)

echo "=== [E5] [1/6] Stage 1: local_field (n_sites=16, local_channels=2 -> d_latent=32, comfortably above true D_KY=11.3) + local_mlp propagator, Section 224's exact regularizer recipe, on Lorenz-96 N=64/F=4.2 ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --nx 64 --dt-snap 0.1 \
  --encoder local_field --local-field-n-sites 16 --local-field-channels 2 \
  --aux-backbone local_mlp --mode markovian \
  --aux-n-tokens 16 --attn-window 3 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
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

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$PROP_PATH')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
print('[$LABEL] d_latent:', d_latent)

with h5py.File('$DATASET','r') as f:
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

with h5py.File('$DATASET','r') as f:
    traj = torch.tensor(f['trajectories'][3], dtype=torch.float32)
with torch.no_grad():
    z01 = ae.encode(traj[:2]).numpy()
res = lyapunov_spectrum_latent_propagator(
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=2000, qr_every=10,
    dt_snap=0.1, warmup_steps=200, seed=0, max_abs_state=1e3,
)
print('[$LABEL] lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('[$LABEL] D_KY:', res.kaplan_yorke_dimension, ' D_KY/N:', res.kaplan_yorke_dimension/64.0)
except Exception as e:
    print('[$LABEL] D_KY: could not bracket --', e)
print('[$LABEL] (true L96 N=64/F=4.2 reference: lambda1=0.080, n_positive=5/64, D_KY=11.3, D_KY/N=0.1766)')
"
}

echo "=== [E5] [2/6] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== [E5] [3/6] Stage 2: warm-started, k_max=12, 300 epochs (matches KS's own Section 224 schedule; L96 sections historically used 5x fewer epochs for a faster-sampled dataset -- this dataset's own snapshot count should be checked before assuming that shortcut applies here too, so the full schedule is kept as the safe default) ==="
STAGE2_TAG="${TAG}_warmstart_k12_300ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --dataset "$DATASET" \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [E5] [4/6] Stage 2 diagnostics ==="
_diagnose "$STAGE2_PROP" "stage2" > artifacts/logs/lyapunov_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/lyapunov_${STAGE2_TAG}.log

echo "=== [E5] [5/6] IF Stage 2 shows bounded, genuinely chaotic dynamics (D_KY comfortably positive, not collapsed to ~0): N-transfer check, N=64 -> N=128, zero retraining ==="
echo "    (this step is only meaningful if [4/6] did not collapse -- read that output before trusting this one)"
mamba run -n da_env python -c "
import numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint_resized, load_propagator_checkpoint_resized
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

old_scale = 2
ae, ae_cfg, _ = load_autoencoder_checkpoint_resized('$AE', new_n_sites=16*old_scale, new_NX=64*old_scale)
d_latent = ae_cfg.n_sites * ae_cfg.local_channels
prop, prop_cfg, _ = load_propagator_checkpoint_resized('$STAGE2_PROP', new_d_latent=d_latent, new_n_tokens=16*old_scale)
ae.eval(); prop.eval()
print('[E5 N-transfer] resized to N=128, d_latent=', d_latent, '-- ZERO retraining')

from ks_latent.config import Lorenz96Config
from ks_latent.solver.lorenz96 import integrate as l96_integrate, spinup as l96_spinup
cfg = Lorenz96Config(N=128, F=4.2, dt=0.01, snapshot_every=10, spinup_time=200.0, seed=0)
u0 = l96_spinup(cfg, np.random.default_rng(0))
truth = l96_integrate(u0, cfg, n_steps=2000*cfg.snapshot_every)
with torch.no_grad():
    z01 = ae.encode(torch.tensor(truth[:2], dtype=torch.float32)).numpy()
res = lyapunov_spectrum_latent_propagator(
    prop, z01, mode='single_state', n_directions=d_latent, n_steps=2000, qr_every=10,
    dt_snap=0.1, warmup_steps=200, seed=0, max_abs_state=1e3,
)
print('[E5 N-transfer] N=128 D_KY:', res.kaplan_yorke_dimension, ' D_KY/N:', res.kaplan_yorke_dimension/128.0)
print('[E5 N-transfer] compare: N=64 trained-size D_KY/N should be close to this ratio if transfer holds')
" > artifacts/logs/e5_ntransfer_N128.log 2>&1 || echo "N-transfer step failed or ks_latent.solver.lorenz96 API differs from assumed -- see artifacts/logs/e5_ntransfer_N128.log"
cat artifacts/logs/e5_ntransfer_N128.log

echo "=== [E5] [6/6] complete ==="
