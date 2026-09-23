#!/bin/zsh
# User-directed 2026-09-07, "Section 103": combines four changes on top of
# Section 102's joint Phase-1/Phase-2 design, all discussed and agreed in
# conversation before launch:
#
#   1. K=15 (up from 8), N_w=32 UNCHANGED (d_latent=2*K=30, up from 16).
#      User: "why are we truncating at 8? that seems arbitrary? why not
#      use the whole spectrum we can but also have a low pass filter,
#      that was the whole reason for the filter." N_w=32's own Nyquist
#      limit caps K at 17 (n_freq = N_w//2+1); K=15 leaves just the top 2
#      modes discarded -- "the whole spectrum we can have at this
#      resolution," not a new N_w increase (deliberately -- see the
#      relaxed-dealiasing discussion below; growing N_w too was
#      considered and set aside to keep this pass's added compute cost
#      isolated to the OTHER three changes below, which already compound
#      substantially on their own).
#      Classical spectral dealiasing (Orszag 3/2 rule, N_w>=3K) is
#      deliberately NOT respected here (3*15=45 > 32) -- discussed
#      explicitly: the rule protects against a FIXED, non-adaptive
#      nonlinearity's aliasing corrupting kept modes, a concern that
#      doesn't transfer cleanly to a TRAINED, smooth MLP nonlinearity
#      (which the low-pass filter already discourages from needing sharp
#      high-k corrections, and which would show measurable training
#      instability if aliasing were actually corrupting things, unlike a
#      classical scheme's silent, uncorrectable error). Treated as an
#      empirical question to watch (via existing diagnostics, e.g. the
#      covariance spectrum step), not a hard constraint to design around --
#      a direct energy-in-discarded-modes diagnostic was discussed but NOT
#      built this pass (flagged as a nice-to-have follow-up).
#      w_lowpass RECALIBRATED at K=15 (was tuned for K=8): verified by
#      direct computation on a fresh, untrained AE + real L=22 data (dt_snap=1
#      dataset): at init, l_recon=1.471, l_lowpass(power=1)=30.289 (ratio
#      20.6x). w_lowpass=0.007 gives a weighted contribution of ~0.212
#      (~14% of l_recon at init) -- same target fraction as Section
#      101/102's calibration (0.003 at K=8 gave ~14% there too).
#
#   2. dt_snap=1.0 (this project's own canonical value, up from Section
#      101/102's 0.2). User: "change delta t back to 1 to discourage
#      collapse to a single point." At dt_snap=0.2, consecutive snapshots
#      are already close together, so a lazy near-identity/contracting
#      map already achieves low prediction error -- little training
#      pressure against exactly the collapse (D_KY=0) observed in every
#      Section 101 propagator. At dt_snap=1.0, consecutive snapshots
#      differ meaningfully, so a trivial contracting solution incurs real,
#      visible loss. New dataset generated this turn (24.8s):
#      artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
#      (snapshot_every=20 at the solver's own dt=0.05 -- literally this
#      project's canonical snapshot_every, no special-casing;
#      trajectory_time=250.0, the canonical value, 250 snapshots/run).
#      k_max=12 at dt_snap=1.0 now spans the FULL canonical 12 physical
#      time units (~0.55 Lyapunov times at this L=22 system's own
#      lambda_1~=0.045), vs. Section 101/102's 2.4 units (~0.11 Lyapunov
#      times) -- a real, substantial improvement in how much of the
#      system's own predictability horizon the rollout curriculum
#      actually covers, not just a parameter change for its own sake.
#
#   3. ode_substeps=3 (both --aux-backbone spectral_pde's Phase 1 AND
#      Phase 2's warm-started continuation, since --init-prop-checkpoint
#      loads architecture -- including ode_substeps -- directly from
#      Phase 1's own checkpoint, so this only needs setting once, in
#      Phase 1). User: "we might consider integrating rk4 multiple times
#      in that time step," directly following from point 2 -- a single
#      big step of size dt_snap=1.0 is 5x coarser than Section 101/102's
#      0.2, and the underlying PDE is genuinely stiff, so sub-stepping
#      keeps the effective per-substep size reasonable. User-directed
#      compute-cost trim (this session's AskUserQuestion): ode_substeps=3
#      (not 5 -- effective per-substep size 1.0/3~=0.33, still ~1.65x
#      coarser than Section 101/102's own 0.2, an accepted compute/
#      accuracy tradeoff for this pass) -- 3x the field() (MLP)
#      evaluations per rollout step for rk4/etdrk4 vs Section 102's
#      ode_substeps=1 (euler always ignores ode_substeps, a single full
#      step regardless -- see _SpectralPDEDeltaBody's docstring; unaffected
#      by this change, which is exactly why it's the cheapest of the three
#      and run first per the user's explicit ordering).
#
#   4. w_pred=1.5 in Stage 1 (Stage1TrainingConfig.w_pred, default 0.5).
#      User: "also increase w_pred to 1.5 in stage 1." Directly reinforces
#      Section 12's own joint-training fix purpose: weight L_pred (the
#      term pressuring the encoder to co-adapt with the REAL propagator,
#      now that Stage 1 trains it directly instead of a throwaway generic
#      one) more heavily relative to pure reconstruction.
#
# Order: euler, then rk4, then etdrk4 -- user-directed (cheapest first,
# given ode_substeps doesn't affect euler's cost at all).
#
# Verified via a real smoke run before launching (4-epoch Phase 1 + 2-epoch
# Phase 2, K=15/dt_snap=1.0/ode_substeps=3/w_pred=1.5 all combined, euler
# integrator): both phases ran cleanly end to end, no errors. Smoke
# artifacts cleaned up before this real launch.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
REG_FLAGS="--w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 --w-lowpass 0.007 --lowpass-power 1.0"
PHASE2_COMMON="--k-max 12 --k-mid 8 --k-warmup-epochs 56 --k-mid-epochs 46 --epochs 80"

for INTEGRATOR in euler rk4 etdrk4; do
  TAG="section103_spectralfield_L22_K15_Nw32_dtsnap1_${INTEGRATOR}"

  echo "=== [$INTEGRATOR 1/3] Phase 1 (JOINT): spectral_field AE (K=15, N_w=32) + REAL spectral_pde aux propagator (--spectral-integrator $INTEGRATOR, ode_substeps=3, --full-propagator hidden=128/n_blocks=3), w_pred=1.5, dt_snap=1.0, --amp, 200 epochs ==="
  mamba run -n da_env python scripts/train_stage1_patched.py \
    --profile full --dataset "$DATASET" --dt-snap 1.0 \
    --encoder spectral_field --d-latent 32 --spectral-K 15 --spectral-L 22.0 \
    --aux-backbone spectral_pde --spectral-integrator "$INTEGRATOR" --ode-substeps 3 --mode markovian \
    --full-propagator --w-pred 1.5 \
    $REG_FLAGS \
    --amp --epochs 200 --checkpoint-every 20 \
    --tag "$TAG" \
    > artifacts/logs/stage1_${TAG}.log 2>&1

  AE=artifacts/stage1_ae_patched_full_${TAG}.pt
  AUX=artifacts/stage1_prop_full_${TAG}.pt

  echo "=== [$INTEGRATOR 2/3] Phase 2 (extend rollout): warm-started from Phase 1's own $INTEGRATOR propagator, k_pred=2 -> k_max=12 (full canonical physical horizon at dt_snap=1.0), --amp, 80 epochs ==="
  STAGE2_TAG="${TAG}_extended"
  mamba run -n da_env python scripts/train_stage2_patched.py \
    --ae-checkpoint "$AE" \
    --init-prop-checkpoint "$AUX" \
    --amp $PHASE2_COMMON \
    --tag "$STAGE2_TAG" \
    > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

  echo "=== [$INTEGRATOR 3/3] Gate 3/4 diagnostics ==="
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

  echo "=== [$INTEGRATOR] complete ==="
done

echo "=== Section 103 complete (euler, rk4, etdrk4; K=15/dt_snap=1.0/ode_substeps=3/w_pred=1.5) ==="
