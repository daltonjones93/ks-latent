#!/bin/zsh
# User-directed 2026-09-13. "seems like the pde from 167 could actually
# be used as a propagator potentially. do you agree? if so we can make
# that 169."
#
# Section 167's pde_head never diverges standalone, its self-spectrum
# mixing check shows no drift toward mode-0 across any of 10 real ICs
# (unlike Section 164's 72-86%% mode-0 concentration), and its fit
# against the trained propagator is decent (R2@k=8=0.74). Worth testing
# directly as an ACTUAL forecasting/DA propagator, not just a closure
# that mimics the trained one -- that IS the actual goal of this whole
# research arc.
#
# NO retraining: pde_head is already saved in the exact same checkpoint
# format every propagator uses (`load_propagator_checkpoint`, same
# `.rollout()`/`.step_one()` interface) -- this section just points the
# EXISTING diagnostic scripts at Section 167's own pde_head checkpoint
# directly, in place of the trained masked_mlp_expand propagator.
#
# Verified directly before writing this script (ad hoc, same commands
# used here): run_da_pff.py accepted the pde_head checkpoint with no
# code changes and ran to completion.
set -e
cd /Users/daltonjones/Documents/latent_DA

SRC_TAG=section167_excludenonconservative
AE=artifacts/stage1_ae_patched_full_${SRC_TAG}.pt
PDEHEAD=artifacts/stage2_pdehead_patched_full_${SRC_TAG}_warmstart_k12_300ep.pt
TAG=section169_pdehead_as_propagator

echo "=== [1/4] Visualization -- pde_head's own free rollout treated as THE propagator ==="
mamba run -n da_env python scripts/visualize_rollout.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PDEHEAD" \
  --rollout-steps 200 --tag "$TAG" \
  > artifacts/logs/visualize_${TAG}.log 2>&1
cat artifacts/logs/visualize_${TAG}.log

echo "=== [2/4] Gate 3 analysis suite (dimension/topology/Lyapunov) on pde_head-as-propagator ==="
mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PDEHEAD" \
  --tag "$TAG" > artifacts/logs/gate3_analysis_${TAG}.log 2>&1
cat artifacts/logs/gate3_analysis_${TAG}.log

echo "=== [3/4] Gate 3 DA/PFF skill on pde_head-as-propagator ==="
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PDEHEAD" \
  --tag "$TAG" > artifacts/logs/gate3_da_${TAG}.log 2>&1
cat artifacts/logs/gate3_da_${TAG}.log

echo "=== [4/4] Gate 4 diagnostics (D1-D9) on pde_head-as-propagator ==="
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PDEHEAD" \
  --tag "$TAG" > artifacts/logs/gate4_diagnostics_${TAG}.log 2>&1

echo "=== Section 169 complete ==="
grep -E "D_KY|n_positive|lambda1" artifacts/logs/gate3_analysis_${TAG}.log || true
grep -E "skill_free_over_da|calibration|rmse_da|rmse_free" artifacts/logs/gate3_da_${TAG}.log || true
grep -E "D3 p-value|D9 smoothness" artifacts/logs/gate4_diagnostics_${TAG}.log || true
