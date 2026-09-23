#!/bin/zsh
# User-directed 2026-09-07, "Section 104": "kill it. I want to try the
# euler mode with the specreal_shape_floor_loss. set it at whatever value
# you think will make a difference, and let's see if we can get something
# that doesn't collapse. also increase the latent space dimension a bit
# and the number of modes for z."
#
# Section 103's rk4 leg was killed mid-Phase-2 (euler's own result already
# in hand: D_KY=0 still, but lambda_1=-0.030, ~15x closer to neutral than
# Section 101's -0.455 -- real progress, not yet over the line). This
# section bundles TWO changes on top of Section 103's euler design at once
# (both explicitly requested together):
#
#   1. N_w=64 (up from 32), K=24 (up from 15) -- "increase the latent
#      space dimension a bit and the number of modes for z." NX=256 stays
#      divisible (patch_size=NX/N_w=256/64=4). d_latent=2*K=48 (close to
#      this project's canonical d_latent=44). K=24 is comfortably below
#      the max K=N_w//2+1=33 this N_w allows.
#
#   2. `--w-shape-floor` now ACTIVE, set aggressively rather than at the
#      gentle ~14%-of-l_recon convention every other regularizer this
#      session used -- user: "set it at whatever value you think will
#      make a difference." Verified by direct computation on this exact
#      K=24/N_w=64 config at init (fresh, untrained AE, real L=22 data):
#      l_recon=1.674, l_shape_floor=17.167 (ratio 10.3x). w_shape_floor=0.1
#      gives a weighted contribution of ~1.72 -- roughly EQUAL to
#      l_recon's own magnitude at init, a deliberately bold weighting (not
#      a gentle nudge) matching the explicit intent to actually test
#      whether this mechanism can prevent collapse, not just nudge at it.
#
# `--w-lowpass` RECALIBRATED for the new K (its raw magnitude scales with
# how many modes are summed AND how high their wavenumbers go -- at K=24
# the highest kept mode's wavenumber is ~6.6 rad/length vs K=15's ~4.0,
# so the raw sum grows much faster than K alone would suggest). Verified:
# at this config's init, l_lowpass=192.834 (ratio 115x l_recon at the OLD
# w_lowpass=0.007 -- would have been wildly dominant). w_lowpass=0.0012
# restores the same ~14%-of-l_recon target fraction every prior section
# used.
#
# Everything else unchanged from Section 103's euler leg: dt_snap=1.0,
# ode_substeps=3, w_pred=1.5, w_var=w_spatial=w_decorr=w_logdet=0, joint
# Phase-1/Phase-2 design (Section 12), same k_max=12/epochs schedule.
# euler ONLY this pass (not rk4/etdrk4) -- user-directed, the specific
# integrator this experiment is about.
#
# Verified via a real smoke run (4-epoch Phase 1 + 2-epoch Phase 2) before
# launching: both phases run cleanly at the new K=24/N_w=64 sizing with
# --w-shape-floor 0.1 active. Smoke artifacts cleaned up before this launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section104_spectralfield_L22_K24_Nw64_shapefloor01_euler

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=24, N_w=64) + REAL spectral_pde aux propagator (euler, ode_substeps=3, --full-propagator hidden=128/n_blocks=3), w_pred=1.5, w_shape_floor=0.1 (AGGRESSIVE), w_lowpass=0.0012 (recalibrated), dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator euler --ode-substeps 3 --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
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

echo "=== [3/3] Phase 2 (extend rollout): warm-started from Phase 1's own euler propagator, k_pred=2 -> k_max=12, --amp, 80 epochs, then Gate 3/4 ==="
STAGE2_TAG="${TAG}_extended"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp --k-max 12 --k-mid 8 --k-warmup-epochs 56 --k-mid-epochs 46 --epochs 80 \
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

echo "=== Section 104 complete ==="
