#!/bin/zsh
# Queues Section 142 (spectral_pde_raw as the sole PRIMARY propagator)
# behind Section 141 to avoid GPU contention.
set -e
cd /Users/daltonjones/Documents/latent_DA

DRIVER_LOG=artifacts/logs/section141_driver.log

echo "=== Waiting for Section 141 to finish -- polling $DRIVER_LOG ==="
while ! grep -q "Section 141 complete" "$DRIVER_LOG" 2>/dev/null; do
  sleep 30
done
echo "=== Section 141 finished at $(date) -- launching Section 142 ==="

zsh scripts/section142_localfield_pdeprimary.sh

echo "=== Queue complete: Section 142 finished at $(date) ==="
