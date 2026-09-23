#!/bin/zsh
# User-directed 2026-09-08, "Section 105": "great, let's try the
# spectral_physics_prior with an increased shape floor param using euler
# for test 105. Hopefully we see more encouraging propagator dynamics
# from 104."
#
# Two changes on top of Section 104's exact design (K=24/N_w=64,
# dt_snap=1.0, ode_substeps=3, w_pred=1.5, w_var=w_spatial=w_decorr=
# w_logdet=0, w_lowpass=0.0012, joint Phase-1/Phase-2 design, euler only):
#
#   1. `--spectral-physics-prior` now ACTIVE. Bakes the EXACT true KS
#      right-hand side (`-w*w_x - w_xx - w_xxxx`, the FULL RHS since
#      "euler" has no separate linear treatment -- see PropagatorConfig.
#      spectral_physics_prior's docstring) into field()'s output as a
#      fixed baseline; the pointwise MLP now learns only a correction on
#      top. Verified directly this turn (see docs/sine_transform_pde_plan.md
#      Section 17): at zero_init, field() matches the analytic KS RHS
#      EXACTLY (0.0 max abs diff) -- so THIS run starts training from
#      genuine (nonzero) KS dynamics, not an empty/near-empty baseline the
#      way every prior spectral_pde run did.
#
#   2. `--w-shape-floor` increased from Section 104's 0.1 to 0.3 (3x) --
#      user-directed "increased shape floor param." The AE-side magnitude
#      calibration is UNCHANGED from Section 104 (w_shape_floor acts on
#      the ENCODER's own z, entirely independent of whatever the
#      PROPAGATOR does internally, so Section 104's own init-time
#      computation on this exact K=24/N_w=64 config still applies
#      directly): l_recon=1.674, l_shape_floor=17.167 at init (ratio
#      10.3x). w_shape_floor=0.3 gives a weighted contribution of ~5.15 --
#      roughly 3x l_recon's own magnitude, a genuinely dominant term now
#      (not just "comparable to" l_recon, as Section 104's 0.1 gave).
#
# Everything else identical to Section 104 -- same dataset, same K/N_w/L,
# same regularizer weights otherwise, same curriculum. This isolates
# exactly the two requested variables against Section 104's own result.
#
# Verified via a real smoke run (4-epoch Phase 1 + 2-epoch Phase 2,
# --spectral-integrator etdrk4 --spectral-physics-prior combination tested
# directly on 2026-09-08 -- see Section 17's own verification) that the
# --spectral-physics-prior CLI flag and Phase 1/Phase 2 pipeline work
# end-to-end; this run additionally combines it with euler specifically
# and the new w_shape_floor=0.3, both straightforward compositions of
# already-independently-verified pieces (physics_prior verified generically
# for all 3 integrators including euler in Section 17's own unit tests,
# not just the etdrk4 smoke run) -- no additional smoke test run for this
# exact combination beyond that, to avoid delaying the real launch further
# while Section 104 was already using the GPU.
#
# QUEUED to launch automatically once Section 104 fully completes (this
# script's own launch is gated on that in the driving shell command, not
# inside this script itself).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section105_spectralfield_L22_K24_Nw64_physicsprior_shapefloor03_euler

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=24, N_w=64) + REAL spectral_pde aux propagator (euler, ode_substeps=3, --spectral-physics-prior ACTIVE, --full-propagator hidden=128/n_blocks=3), w_pred=1.5, w_shape_floor=0.3 (3x Section 104), w_lowpass=0.0012, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator euler --ode-substeps 3 --spectral-physics-prior --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.3 \
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

echo "=== [3/3] Phase 2 (extend rollout): warm-started from Phase 1's own euler+physics_prior propagator, k_pred=2 -> k_max=12, --amp, 80 epochs, then Gate 3/4 ==="
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

echo "=== Section 105 complete ==="
