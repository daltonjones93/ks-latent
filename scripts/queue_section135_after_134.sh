#!/bin/zsh
# Queues Section 135 (local_field encoder + masked_mlp_expand) behind
# Section 134 (still training on MPS at launch time) to avoid GPU
# contention -- same practice as Section 98 queuing behind 97.
set -e
cd /Users/daltonjones/Documents/latent_DA

DRIVER_LOG=artifacts/logs/section134_driver.log

echo "=== Waiting for Section 134 to finish -- polling $DRIVER_LOG ==="
while ! grep -q "Section 134 complete" "$DRIVER_LOG" 2>/dev/null; do
  sleep 30
done
echo "=== Section 134 finished at $(date) -- launching Section 135 ==="

zsh scripts/section135_localfield_maskedmlpexpand.sh

echo "=== Queue complete: Section 135 finished at $(date) ==="
