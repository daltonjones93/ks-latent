#!/bin/zsh
# Queues Section 144 (masked_mlp_expand, attn_window=8, properly sized off
# Section 141's own D3 coupling measurement) behind Section 143 to avoid
# GPU contention.
set -e
cd /Users/daltonjones/Documents/latent_DA

DRIVER_LOG=artifacts/logs/section143_driver.log

echo "=== Waiting for Section 143 to finish -- polling $DRIVER_LOG ==="
while ! grep -q "Section 143 complete" "$DRIVER_LOG" 2>/dev/null; do
  sleep 30
done
echo "=== Section 143 finished at $(date) -- launching Section 144 ==="

zsh scripts/section144_localfield_maskedmlpexpand_w8.sh

echo "=== Queue complete: Section 144 finished at $(date) ==="
