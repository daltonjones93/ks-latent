#!/bin/zsh
# Queues Section 152 (chebyshev-basis pde_head) behind Section 151
# (low-derivative-order/high-power polynomial pde_head) to avoid MPS
# contention -- corrects a mistake where both were launched concurrently
# (user: "whoa, don't run both 152 and 151 at once, that's too much for
# the mps").
set -e
cd /Users/daltonjones/Documents/latent_DA

DRIVER_LOG=artifacts/logs/section151_driver.log

echo "=== Waiting for Section 151 to finish -- polling $DRIVER_LOG ==="
while ! grep -q "Section 151 complete" "$DRIVER_LOG" 2>/dev/null; do
  sleep 30
done
echo "=== Section 151 finished at $(date) -- launching Section 152 ==="

zsh scripts/section152_localfield_maskedmlpexpand_chebyshev.sh

echo "=== Queue complete: Section 152 finished at $(date) ==="
