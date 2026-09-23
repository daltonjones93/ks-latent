#!/bin/zsh
# User-directed 2026-09-07, "Section 102": "I don't think this is a good
# idea. I want the encoder and decoder pair to be trained specifically to
# encode data that can be transformed accurately by the rk4/etdrk4/euler
# method. therefore each of those should be present in phase 1 and then
# we can use the same network in phase 2 as we extend the rollout."
#
# Direct correction of Section 101's design flaw: there, Stage 1 trained
# the spectral_field AE against a cheap, GENERIC "mlp"/markovian aux
# propagator (used only for Stage 1's own L_pred regularizer), then Stage
# 2 built a completely FRESH spectral_pde propagator from scratch on top
# of that already-frozen AE -- the encoder never had any pressure to
# produce a z that's actually easy for a spectral-derivative-based
# propagator specifically to work with. Section 101's rk4 result came back
# D_KY=0.0 (n_positive=0, lambda_1=-0.65, a genuine collapse to a fixed
# point) -- consistent with H-PROP, but confounded by this design gap: we
# don't know whether the collapse is intrinsic to the architecture, or
# partly an artifact of training the encoder/decoder in isolation from the
# propagator that would eventually consume its z.
#
# NEW CAPABILITY built this turn to fix this: `AuxPropagatorConfig` (Stage
# 1's own propagator config, previously spectral_pde-incompatible, matching
# "node"/"cnn" precedent of being Stage-2-only) now supports
# backbone="spectral_pde" directly, mirroring every field
# PropagatorConfig already has (spectral_K/spectral_N_w/spectral_L/
# spectral_max_order/spectral_integrator, plus a NEW ode_substeps field
# AuxPropagatorConfig never needed before). `aux_cfg_to_propagator_cfg`
# forwards all of them. train_stage1_patched.py gained --aux-backbone
# spectral_pde plus --spectral-max-order/--spectral-integrator/
# --ode-substeps CLI flags. Verified via a real smoke run: Stage 1 now
# jointly trains the AE + a REAL, full-sized (--full-propagator,
# hidden=128/n_blocks=3) spectral_pde propagator together, and Stage 2
# correctly warm-starts from that exact checkpoint via
# --init-prop-checkpoint to extend the rollout -- exactly the workflow
# requested.
#
# Design: THREE FULL joint pipelines (one per integrator), each:
#   Phase 1 (joint): spectral_field AE + REAL spectral_pde aux propagator
#     (--full-propagator, hidden=128/n_blocks=3, matching Stage 2's own
#     sizing exactly so warm-starting is a like-for-like continuation, not
#     a resize) trained TOGETHER. Stage 1's OWN L_pred rollout stays at
#     its default k_pred=2 (no ramp) -- short and cheap, a regularizing
#     joint-adaptation signal, not the main long-horizon training (that is
#     still Phase 2's job, matching the user's own "then we can use the
#     same network in phase 2 as we extend the rollout" framing).
#   Phase 2 (extend rollout): warm-start via --init-prop-checkpoint from
#     Phase 1's own saved propagator, ramp k_pred=2 -> k_max=12 over 80
#     epochs (identical curriculum to Section 101, for comparability).
# Same L=22 dataset, same K=8/N_w=32/L=22.0, same regularizers
# (w_var=w_spatial=w_decorr=w_logdet=0, w_lowpass=0.003) as Section 101 --
# ONLY the joint-vs-frozen encoder question differs between the two
# sections, isolating exactly that variable.
#
# Gate 3/4 diagnostics run on all three resulting propagators, same
# scripts/flags as Section 101 (--L 22.0/--dt-snap 0.2, the same L=100
# hardcoding fix from that section already in place).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap02.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
REG_FLAGS="--w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 --w-lowpass 0.003 --lowpass-power 1.0"
PHASE2_COMMON="--k-max 12 --k-mid 8 --k-warmup-epochs 56 --k-mid-epochs 46 --epochs 80"

for INTEGRATOR in rk4 etdrk4 euler; do
  TAG="section102_spectralfield_L22_K8_Nw32_joint_${INTEGRATOR}"

  echo "=== [$INTEGRATOR 1/3] Phase 1 (JOINT): spectral_field AE + REAL spectral_pde aux propagator (--spectral-integrator $INTEGRATOR, --full-propagator hidden=128/n_blocks=3), --amp, 200 epochs ==="
  mamba run -n da_env python scripts/train_stage1_patched.py \
    --profile full --dataset "$DATASET" --dt-snap 0.2 \
    --encoder spectral_field --d-latent 32 --spectral-K 8 --spectral-L 22.0 \
    --aux-backbone spectral_pde --spectral-integrator "$INTEGRATOR" --ode-substeps 1 --mode markovian \
    --full-propagator \
    $REG_FLAGS \
    --amp --epochs 200 --checkpoint-every 20 \
    --tag "$TAG" \
    > artifacts/logs/stage1_${TAG}.log 2>&1

  AE=artifacts/stage1_ae_patched_full_${TAG}.pt
  AUX=artifacts/stage1_prop_full_${TAG}.pt

  echo "=== [$INTEGRATOR 2/3] Phase 2 (extend rollout): warm-started from Phase 1's own $INTEGRATOR propagator, k_pred=2 -> k_max=12, --amp, 80 epochs ==="
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
    --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --dt-snap 0.2 \
    --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1 &
  mamba run -n da_env python scripts/run_da_pff.py \
    --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
    --dt-snap 0.2 --L 22.0 \
    --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1 &
  mamba run -n da_env python scripts/run_diagnostics.py \
    --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
    --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --L 22.0 \
    --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1 &
  wait

  echo "=== [$INTEGRATOR] complete ==="
done

echo "=== Section 102 complete (all three integrators, joint-trained design) ==="
