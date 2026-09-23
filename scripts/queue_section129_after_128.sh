#!/bin/zsh
# User-directed 2026-09-10: "queue it after stage 1 and 2 finishes for 128
# and diagnostics are run." Waits for Section 128's driver
# (scripts/section128_vit98_maskedmlpexpand_w3_x3.sh, already running in
# the background) to print its own completion line, THEN runs Gate 3/4
# diagnostics on Section 128's finished checkpoint (same pattern as
# scripts/section52_gate34.sh), THEN launches Section 129
# (scripts/section129_vitglobal_maskedmlpexpand_w3_x3.sh). One background
# job for the whole chain so it can run unattended for however long 128's
# remaining training + diagnostics + 129's own training take.
set -e
cd /Users/daltonjones/Documents/latent_DA

DRIVER_LOG=artifacts/logs/section128_driver.log

echo "=== Waiting for Section 128 (Stage 1 + Stage 2) to finish -- polling $DRIVER_LOG ==="
while ! grep -q "Section 128 complete" "$DRIVER_LOG" 2>/dev/null; do
  sleep 30
done
echo "=== Section 128 training finished at $(date) -- running Gate 3/4 diagnostics ==="

TAG=section128_vitonly_dmodel56_tokenmlp_propmaskedmlpexpand_w3_x3_200ep_warmstart_k12_300ep
AE=artifacts/stage1_ae_patched_full_section128_vitonly_dmodel56_tokenmlp_propmaskedmlpexpand_w3_x3_200ep.pt
PROP=artifacts/stage2_prop_patched_full_${TAG}.pt

mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$TAG" > artifacts/logs/gate3_analysis_${TAG}.log 2>&1 &
P1=$!
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$TAG" > artifacts/logs/gate3_da_${TAG}.log 2>&1 &
P2=$!
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --tag "$TAG" > artifacts/logs/gate4_diagnostics_${TAG}.log 2>&1 &
P3=$!
wait $P1 $P2 $P3

echo "=== Section 128 Gate 3/4 complete at $(date) ==="
grep -E "D_KY|n_positive|lambda1|skill_free_over_da" artifacts/logs/gate3_analysis_${TAG}.log || true
grep -E "D3 p-value|D4 verdict|D6 p-value|D7 p-value|D8 p-value" artifacts/logs/gate4_diagnostics_${TAG}.log || true

echo "=== Launching Section 129 (ViT global encoder/decoder, everything else identical to 128) at $(date) ==="
zsh scripts/section129_vitglobal_maskedmlpexpand_w3_x3.sh

echo "=== Queue complete: Section 129 finished at $(date) ==="
