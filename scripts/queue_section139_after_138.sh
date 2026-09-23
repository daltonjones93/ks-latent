#!/bin/zsh
# Queues Section 139 behind Section 138 (still running both 136 and 137
# distillations) to avoid GPU contention.
set -e
cd /Users/daltonjones/Documents/latent_DA

DRIVER_LOG=artifacts/logs/section138_driver.log

echo "=== Waiting for Section 138 to finish -- polling $DRIVER_LOG ==="
while ! grep -q "Section 138 complete" "$DRIVER_LOG" 2>/dev/null; do
  sleep 30
done
echo "=== Section 138 finished at $(date) -- launching Section 139 ==="

zsh scripts/section139_localfield_jointpde.sh

echo "=== Queue complete: Section 139 finished at $(date) ==="
