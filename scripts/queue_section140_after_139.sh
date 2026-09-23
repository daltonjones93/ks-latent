#!/bin/zsh
# Queues Section 140 (fully mutual pde_distill, Stage 1 AND Stage 2)
# behind Section 139 (Stage-2-only mutual) to avoid GPU contention.
set -e
cd /Users/daltonjones/Documents/latent_DA

DRIVER_LOG=artifacts/logs/section139_driver.log

echo "=== Waiting for Section 139 to finish -- polling $DRIVER_LOG ==="
while ! grep -q "Section 139 complete" "$DRIVER_LOG" 2>/dev/null; do
  sleep 30
done
echo "=== Section 139 finished at $(date) -- launching Section 140 ==="

zsh scripts/section140_localfield_jointpde_mutual.sh

echo "=== Queue complete: Section 140 finished at $(date) ==="
