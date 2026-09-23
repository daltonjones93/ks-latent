#!/bin/zsh
# User-directed 2026-09-09, "take stock and think about a different
# direction" (after Section 119 -- etdrk4 alone, no physics_prior --
# also collapsed to a fixed point, same as every learned-polynomial
# variant tried: Sections 111-119 across K=5..24, degree 1-3, various
# normalization/regularizer combinations, poly_stable_leading, euler vs
# etdrk4 -- every single one either collapsed or diverged).
#
# TAKING STOCK: every prior attempt asked the optimizer to DISCOVER
# chaotic dynamics from scratch via the learned polynomial coefficients.
# "Contract to a fixed point" is the cheap, always-available answer for
# that optimization problem (H-PROP), and nothing tried so far has
# changed that incentive.
#
# NEW DIRECTION: `spectral_physics_prior` (built 2026-09-08, verified at
# zero-init, but NEVER SUCCESSFULLY RUN as a real experiment until now --
# a prior attempt, Section 105, used euler and went straight to NaN at
# epoch 0; root-caused directly this session: euler is a naive explicit
# step, and the EXACT KS w_xxxx term is numerically STIFF -- at K=24,
# L=22, the explicit-Euler stability limit is dt<0.0009, but the actual
# substep used is 0.333, ~370x too large -- a textbook CFL violation,
# nothing to do with training dynamics at all). This run uses etdrk4
# instead, which handles the stiff linear part via the SAME exact
# exponential/contour-integral formula the real KS solver itself uses --
# specifically designed to avoid exactly this failure mode.
#
# The mechanism: physics_prior bakes in the EXACT analytic KS right-hand
# side (`w_t = -w*w_x - w_xx - w_xxxx`, or for etdrk4, just the
# `-w*w_x` nonlinear part -- the linear `-w_xx-w_xxxx` is already handled
# exactly via `exp(dt*(k^2-k^4))`) as a FIXED, non-learned baseline; the
# learned polynomial only adds a residual CORRECTION on top, zero at
# init. This changes the fundamental incentive: the "lazy" solution
# (correction ~= 0) is no longer a collapsed fixed point -- it's the
# EXACT true KS equation, which is genuinely chaotic BY CONSTRUCTION.
# Training only needs to preserve that property while adapting to
# whatever the encoder's z actually represents, not discover chaos from
# scratch.
#
# VERIFIED DIRECTLY before committing any training compute (the single
# most important check): built a completely FRESH, zero-init,
# physics_prior+etdrk4 propagator (literally zero propagator training)
# and paired it with Section 104's own already-trained AE checkpoint.
# Result: genuine spatiotemporal structure -- persistent traveling-wave-
# like moving boundaries for the full 200-step rollout, in BOTH physical
# and latent space -- qualitatively completely different from every flat,
# static collapse seen in Sections 112-119. Some amplitude saturation
# was visible (expected -- Section 104's encoder was never trained
# alongside this exact physics), motivating this real JOINT training run
# so the encoder can co-adapt.
#
# Verified via real smoke tests too: Phase 1 trains cleanly
# (val_recon_final=0.049 after 3 epochs, no NaN); a 200-step rollout
# completed without diverging; Phase 2 continuation smoke test held
# val_kmax_mse~0.093 across 5 epochs -- far above the suspiciously-tiny
# values (~0.0005-0.005) every collapsed run in this arc showed at the
# same stage.
#
# Recipe: K=24, N_w=64 (Section 104's own sizing -- no longer avoiding
# large K for instability reasons, since physics_prior+etdrk4 handles the
# stiff linear part EXACTLY regardless of K), degree=2 (sufficient for
# -w*w_x, KS's own nonlinear term), max_term_order=5, standard
# normalization (poly_norm_power=1.0, no need to weaken it -- the
# correction should be small), w_shape_floor=0.1 (Section 104's own
# baseline, NOT Section 105's 0.3 -- no reason to escalate it here), no
# w_logdet_physical/w_var_physical/etc this round (isolating
# physics_prior+etdrk4's own effect cleanly, matching this session's
# established discipline of not changing multiple things at once).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section120_spectralfield_L22_K24_Nw64_physicsprior_etdrk4

echo "=== [1/3] Phase 1 (JOINT): spectral_field AE (K=24, N_w=64) + REAL spectral_pde aux propagator (ETDRK4, ode_substeps=3, PHYSICS_PRIOR ACTIVE (exact KS RHS baked in, learned polynomial correction only, degree=2, max_order=4, max_term_order=5), --full-propagator), w_pred=1.5, w_shape_floor=0.1, w_lowpass=0.0012, w_lowpass_rollout=0.02, dt_snap=1.0, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder spectral_field --d-latent 64 --spectral-K 24 --spectral-L 22.0 \
  --aux-backbone spectral_pde --spectral-integrator etdrk4 --ode-substeps 3 \
  --spectral-physics-prior \
  --spectral-field-kind polynomial --spectral-poly-degree 2 --spectral-poly-max-term-order 5 \
  --mode markovian \
  --full-propagator --w-pred 1.5 \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-lowpass 0.0012 --lowpass-power 1.0 --w-shape-floor 0.1 \
  --w-lowpass-rollout 0.02 --lowpass-rollout-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

echo "=== [2/3] Phase 2 (extend rollout): warm-started from Phase 1's own propagator, k_pred=2 -> k_max=12, w_lowpass_rollout=1.0, --amp, 80 epochs, then Gate 3/4 ==="
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

echo "=== Section 120 complete ==="
