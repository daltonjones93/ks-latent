#!/bin/zsh
# User-directed 2026-09-08, "Section 107": builds on the full pde_head
# hybrid arc from this session (docs/sine_transform_pde_plan.md §19-22):
#   - §19: pde_head distillation, joint with Stage 1 (Stage-1-only at
#     first, spectral_field encoder only, target DETACHED so aux stays
#     protected while the encoder co-adapts).
#   - §20: generalized to backbone="spectral_pde_raw" -- pde_head takes
#     its OWN self-FFT of ANY encoder's raw z, so it works with Section
#     95's actual architecture (vit_fourier_hybrid + mlp/markovian,
#     d_latent=44), not just a dedicated spectral_field encoder.
#   - §21: pde_head continues into Stage 2 (two independent options,
#     A=safe single-step-at-every-position, B=risky autoregressive chain
#     through pde_head itself), MUTUAL by construction there (Stage 2 has
#     no encoder to shape, so a detached target would regularize nothing).
#   - §22: Stage 1's OWN pde_head loss can now ALSO be made mutual
#     (`--pde-mutual`), user-directed: "I think the loss should be mutual
#     for stage 1 training too. We always want the propagator to have
#     dynamics that can be easily modeled by the pde_head right?" --
#     agreed in principle (gradient must reach aux directly for aux's
#     OWN dynamics to actually become more PDE-describable, not just an
#     encoder that could support one), with a real, named risk: this is
#     the same "pressure toward simplicity" mechanism behind this
#     project's H-PROP fixed-point-collapse finding.
#
# This section runs the whole arc for real, for the first time: Section
# 95's exact architecture, trained FROM SCRATCH (not warm-started, per
# user direction), with pde_head mutual in BOTH phases.
#
# ============================================================
# NEW physical domain: L=40 (user-directed). No prior Gate-1-style D_KY
# benchmark exists for L=40 in this repo (only L=22, D_KY~5.2-5.6, and
# L=100, D_KY~21-24, are established -- see project_ks_true_dky_benchmark
# memory, itself L=100-specific). Physical intuition (same mechanism as
# docs/sine_transform_pde_plan.md §18's L=22 discussion): KS's linear
# instability band is 0<k<1, k_m=2*pi*m/L, so L=40 admits roughly
# L/(2*pi)~6.4 unstable wavenumbers (m=1..6, k up to ~0.94) -- meaningfully
# more than L=22's 3, meaningfully fewer than L=100's ~16. No number is
# asserted as ground truth here; Gate 3's own D_KY on this run IS the
# measurement.
#
# New datasets generated this session specifically for this section:
#   artifacts/datasets/stage1_trajectories_L40_dtsnap1.h5 (KSConfig(L=40,
#     NX=256, dt=0.05, snapshot_every=20 [dt_snap=1.0], spinup_time=500,
#     seed=0), n_train=50/n_val=10, trajectory_time=250 -- same convention
#     as the canonical L=100 dataset train_stage1_patched.py would have
#     auto-generated, generated explicitly here instead since that
#     auto-gen path hardcodes L=100.0).
#   artifacts/datasets/attractor_points_L40.h5 (same KSConfig, n_runs=
#     10000/spinup_discard_snapshots=100/post_spinup_time=50 -- function
#     defaults, same convention as the L=22 points dataset).
#
# ============================================================
# Architecture: IDENTICAL to Section 95
# (scripts/section95_wspatial03_wsmooth0015.sh) -- vit_fourier_hybrid
# encoder (d_model=64, fourier_hidden=95, fourier_blocks=2, kind=ifft,
# enc_out_modes=8, dec_fno_modes=12, pool/dec_pool=token_mlp,
# token_mlp_reduction=8/hidden=128, attn_window=4, token_window=16,
# pos_encoding=linear), d_latent=44 (default), aux_backbone=mlp/markovian,
# --full-propagator (hidden=128/n_blocks=3). Regularizers UNCHANGED from
# 95: w_decorr=0, w_var=0.02, w_spatial=0.03 (signed), w_var_floor=0,
# w_logdet=0.008, w_smooth=0.0015.
#
# pde_head (NEW): backbone=spectral_pde_raw, K=23 (=d_latent//2+1, no
# truncation), L_pde=44 (=d_latent -- this is pde_head's OWN self-FFT ring
# length for treating the LATENT INDEX as space, entirely independent of
# the PHYSICAL KS domain L=40 above -- do not confuse the two), integrator
# euler (etdrk4 proven harmful for this backbone, §20), hidden=128/
# n_blocks=3 (same capacity as aux).
#
# Weight calibration (real, computed 2026-09-08 against this exact
# architecture, canonical L=100 data as a magnitude proxy -- Section 95's
# own init-time scale): l_pred=1.670 (weighted by w_pred=0.5 -> 0.835),
# raw l_pde_distill (mutual formula, both nets non-zero-init) = 0.397,
# ratio to weighted l_pred ~48%. Targeting the GENTLE ~14%-of-primary
# convention this project uses for a first test of something with a named
# collapse risk (not the "aggressive" convention used for detached-mode
# w_shape_floor etc.): w_pde_distill = 0.3 (both phases).
#
# Phase 2: reuses Section 95's own exact schedule (--epochs 300 --k-max 12
# --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175, from
# scripts/section95_wspatial03_wsmooth0015.sh). pde_head CONTINUES via
# --init-pdehead-checkpoint, option A only this pass (--w-pde-distill 0.3,
# same weight as Phase 1) -- option B (--w-pde-rollout, the risky
# autoregressive-through-pde_head approach) deliberately OFF for this
# FIRST launch: L=40, Stage-1-mutual, and Phase-2-continuation are already
# three simultaneously-new things; adding option B on top would make any
# problem hard to attribute. Explicitly queued as the next iteration once
# this baseline's behavior is understood (user: "we can iterate").
#
# Verified via real smoke runs before launching (2026-09-08): --pde-mutual
# wired correctly (tests/unit/test_pde_head_distillation.py::
# test_aux_receives_nonzero_gradient_when_mutual); full CLI round-trip
# (Phase 1 --pde-distill --pde-mutual -> Phase 2 --init-pdehead-checkpoint
# --w-pde-distill, both options A+B) trains cleanly, no NaN, on a small
# real run. Full test suite green (617+ passed) before this launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L40_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L40.h5
TAG=section107_vitfourierhybrid_L40_pdehead_mutual

echo "=== [1/3] Phase 1 (JOINT, MUTUAL pde_head): vit_fourier_hybrid AE + mlp/markovian aux (full-propagator), w_pde_distill=0.3 --pde-mutual, L=40, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 \
  --encoder vit_fourier_hybrid --aux-backbone mlp --mode markovian \
  --d-model 64 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --vit-fourier-fourier-hidden 95 --vit-fourier-fourier-blocks 2 --vit-fourier-kind ifft \
  --vit-fourier-enc-out-modes 8 \
  --fourier-mlp-dec-fno-modes 12 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.03 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.0015 \
  --pde-distill --w-pde-distill 0.3 --pde-mutual --pde-hidden 128 --pde-n-blocks 3 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/4] Phase 2 (extend rollout, pde_head continues, option A only): k_max=12, 300 epochs (Section 95's own schedule), --amp ==="
STAGE2_TAG="${TAG}_extended"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --init-pdehead-checkpoint "$PDEHEAD" --w-pde-distill 0.3 \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt
PDEHEAD2=artifacts/stage2_pdehead_patched_full_${STAGE2_TAG}.pt

echo "=== [3/4] Phase 3 (docs/sine_transform_pde_plan.md §23): refine pde_head with the propagator FROZEN -- no more collapse risk to protect against, so both options run at a more aggressive weight (1.0 each) than Phases 1-2's 0.3. 100 epochs (pde_head-only training converges fast; Gate 3/4 below evaluates Phase 2's OWN propagator, unaffected by this phase) ==="
PHASE3_TAG="${TAG}_phase3refine"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$PROP" \
  --init-pdehead-checkpoint "$PDEHEAD2" --freeze-propagator \
  --w-pde-distill 1.0 --w-pde-rollout 1.0 \
  --amp --epochs 100 --k-max 12 --k-warmup-epochs 1 \
  --tag "$PHASE3_TAG" \
  > artifacts/logs/stage2_${PHASE3_TAG}.log 2>&1

echo "=== [4/4] Gate 3/4 (on Phase 2's OWN propagator -- Phase 3 only refines pde_head, does not change this) ==="
mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --dt-snap 1.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1 &
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dt-snap 1.0 --L 40.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1 &
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --L 40.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1 &
wait

echo "=== Section 107 complete ==="
