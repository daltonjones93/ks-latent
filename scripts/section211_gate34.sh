#!/bin/zsh
# User-directed 2026-09-24: "Please run the full gates 3, 4 on 211" --
# Section 211's Stage 2 checkpoint (local_field encoder + global mlp
# propagator, Section 52's regularizer recipe) is the first genuinely
# validated local_field checkpoint in this whole investigation:
# D_KY=23.04, lambda1=0.0913, n_positive=13/96 (target [21,24],
# lambda1<=0.1), max|z| genuinely bounded across a full 2000-step
# standalone rollout. This runs the brief's own Gate 3 (analysis suite:
# dimension estimators, Lyapunov, topology) + Gate 3 DA (NAT-PFF cycling,
# --localizer none, the honest no-localization baseline) + Gate 4
# (D1-D5 structure diagnostics, including D3's own post-hoc bandedness
# p-value -- the same statistic --w-jacobian-bandedness, Section 214,
# was built to optimize against) on this checkpoint, matching Section
# 110/52's own established three-way parallel pattern.
set -e
cd /Users/daltonjones/Documents/latent_DA

AE=artifacts/stage1_ae_patched_full_section211_ks_localfield_section52_regs.pt
PROP=artifacts/stage2_prop_patched_full_section211_ks_localfield_section52_regs_warmstart_k12_300ep.pt
DATASET=artifacts/datasets/stage1_trajectories_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points.h5
TAG=section211_ks_localfield_section52_regs_warmstart_k12_300ep

echo "=== Gate 3/4: ${TAG} ==="
mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --dt-snap 1.0 \
  --tag "$TAG" > artifacts/logs/gate3_analysis_${TAG}.log 2>&1 &
P1=$!
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dt-snap 1.0 --L 100.0 --localizer none \
  --tag "$TAG" > artifacts/logs/gate3_da_${TAG}.log 2>&1 &
P2=$!
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --L 100.0 \
  --tag "$TAG" > artifacts/logs/gate4_diagnostics_${TAG}.log 2>&1 &
P3=$!
wait $P1 $P2 $P3

echo "=== Section 211 Gate 3/4 complete ==="
