#!/bin/zsh
# User-directed 2026-09-08, "Section 106": "run 106 as a repeat of 104
# with a somewhat aggressive w_lowpass_rollout term. Let's see if we can
# get D_KY to be closer to 5-6." Corrected immediately after: "wait,
# please don't reuse 104's phase 1 checkpoints, we need to retrain phase 1
# with a lowpass regularization term as well."
#
# Context: Section 104 (K=24, N_w=64, w_shape_floor=0.1, euler) got the
# first genuine positive Lyapunov result (D_KY=22.0, lambda_1=+0.079,
# n_positive=11) -- but this repo's own Gate 1 L=22 benchmark
# (tests/replication/test_gate1_kaplan_yorke.py::test_L22_lyapunov) puts
# the TRUE L=22 attractor's D_KY in [5.2, 5.6] (D_KY~22 is the canonical
# L=100 answer -- see project_ks_true_dky_benchmark memory). The
# irfft(z)-space latent Hovmoller (added this session per user request,
# scripts/visualize_rollout.py's plot_latent_hovmoller) confirmed directly
# in w-space why: the true attractor shows smooth, broad traveling
# structure, while Section 104's free rollout develops persistent narrow
# high-wavenumber streaks the true data never has -- inflating D_KY past
# the physically-correct value while remaining genuinely bounded chaos
# (error growth saturates cleanly near sqrt(2)), not a numerical artifact.
#
# Fix under test: `w_lowpass_rollout` (new this session, both stages --
# see docs/sine_transform_pde_plan.md), the Stage-1 AND Stage-2 analogue
# of `w_lowpass`, but applied to the PROPAGATOR's own rolled-out z_pred
# rather than the encoder's direct z. Active in BOTH phases this time
# (Section 106's first draft only added it to Phase 2, reusing Section
# 104's Phase-1 checkpoints -- user explicitly rejected that: Phase 1
# must retrain from scratch with the term active too, so the encoder and
# the REAL full-sized propagator can co-adapt to the constraint from the
# start rather than a Phase-2-only term fighting an already-fixed Phase-1
# representation).
#
# Weight calibration (computed directly, 2026-09-08, real L=22 data, this
# exact K=24/N_w=64 config):
#   Phase 1 (fresh init, zero_init propagator, k_pred=2 short rollout):
#     l_recon=1.482, l_pred=1.467, raw l_lowpass_rollout(power=1)=109.56
#     (~75x either). w_lowpass_rollout=0.015 gives a weighted contribution
#     of ~1.64 -- roughly EQUAL to l_recon/l_pred's own init magnitude,
#     matching the same "aggressive, not gentle-14%" convention Section
#     104's own w_shape_floor=0.1 used (also ~1x l_recon at init).
#   Phase 2 (against Section 104's own extended, already-trained
#     checkpoint's real 12-step rollout, as a proxy -- Section 106's Phase
#     2 will warm-start from a DIFFERENT, lowpass-regularized Phase 1, but
#     this remains a reasonable ballpark): l_latent(l2)=0.087, raw
#     l_lowpass_rollout(power=1)=0.183 (2.09x). w_lowpass_rollout=1.0
#     gives a weighted contribution of ~2x l_latent -- dominant, not a
#     gentle nudge, without being so extreme (10x+) it destroys the
#     primary tracking loss outright.
#
# Everything else identical to Section 104's own recipe: K=24, N_w=64,
# w_shape_floor=0.1 (unchanged), w_lowpass=0.0012 (unchanged, still
# constrains the encoder's own direct z as before), euler, ode_substeps=3,
# w_pred=1.5, dt_snap=1.0, joint Phase-1/Phase-2 design, k_max=12/k_mid=8/
# k_warmup_epochs=56/k_mid_epochs=46/epochs=80 for Phase 2.
#
# Verified via real smoke runs before launching (2026-09-08): (a)
# --w-lowpass-rollout wired correctly into train_stage1_patched.py (fresh
# spectral_field AE + full spectral_pde propagator, real forward pass, no
# NaN/crash -- confirmed via the calibration script itself, which runs the
# exact same construction); (b) --w-lowpass-rollout wired correctly into
# train_stage2_patched.py against a real checkpoint (2-epoch run, no NaN,
# loss decreased normally).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section106_spectralfield_L22_K24_Nw64_shapefloor01_lowpassrollout_euler

echo "=== [1/3] Phase 1 (JOINT, RETRAINED FROM SCRATCH): spectral_field AE (K=24, N_w=64) + REAL spectral_pde aux propagator (euler, ode_substeps=3, --full-propagator hidden=128/n_blocks=3), w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.015 (NEW), dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator euler --ode-substeps 3 --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
  --w-lowpass-rollout 0.015 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Latent covariance spectrum ==="
mamba run -n da_env python -c "
import h5py, numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint

with h5py.File('$DATASET','r') as f:
    traj = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj.shape
ae, cfg, _ = load_autoencoder_checkpoint('$AE')
ae.eval()
with torch.no_grad():
    z = ae.encode(traj.reshape(n*T, NX))
z_np = z.numpy()
cov = np.cov(z_np, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print('spectrum: min_eig=%.4e  cond#=%.4e  top_eig=%.3f' % (eig[-1], eig[0]/eig[-1], eig[0]))
print('participation ratio: %.3f (out of d_latent=%d)' % (eig.sum()**2 / (eig**2).sum(), cfg.d_latent))
print('per-mode variance (real modes 0..K-1, imag modes K..2K-1):')
print(np.array2string(z_np.var(axis=0), precision=4))
" > artifacts/logs/spectrum_${TAG}.log 2>&1
cat artifacts/logs/spectrum_${TAG}.log

echo "=== [3/3] Phase 2 (extend rollout): warm-started from THIS section's own Phase-1 euler propagator, k_pred=2 -> k_max=12, w_lowpass_rollout=1.0 (NEW, kept active), --amp, 80 epochs, then Gate 3/4 ==="
STAGE2_TAG="${TAG}_extended"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp --k-max 12 --k-mid 8 --k-warmup-epochs 56 --k-mid-epochs 46 --epochs 80 \
  --w-lowpass-rollout 1.0 --lowpass-rollout-power 1.0 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt
mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --dt-snap 1.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1 &
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dt-snap 1.0 --L 22.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1 &
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --L 22.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1 &
wait

echo "=== Section 106 complete ==="
