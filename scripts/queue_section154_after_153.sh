#!/bin/zsh
# Queues Section 154 (w_pde_distill_real -- pde_head trained against real
# encoded ground-truth transitions, not the propagator's own output)
# behind Section 153 (L1 penalty on linear pde coefficients) to avoid MPS
# contention.
set -e
cd /Users/daltonjones/Documents/latent_DA

DRIVER_LOG=artifacts/logs/section153_driver.log

echo "=== Waiting for Section 153 to finish -- polling $DRIVER_LOG ==="
while ! grep -q "Section 153 complete" "$DRIVER_LOG" 2>/dev/null; do
  sleep 30
done
echo "=== Section 153 finished at $(date) -- launching Section 154 ==="

zsh scripts/section154_localfield_maskedmlpexpand_realtargetdistill.sh

echo "=== Queue complete: Section 154 finished at $(date) ==="
