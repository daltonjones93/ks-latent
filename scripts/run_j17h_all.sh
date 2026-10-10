#!/bin/bash
# J17H driver: wait for the offline fits, then run the three damping-order levels in parallel.
cd "$(dirname "$0")/.."
while pgrep -f exp_j17h_fit.py > /dev/null; do sleep 30; done
for p in 5 6 8; do scripts/run_j17h.sh $p & done; wait
echo "$(date '+%F %T') J17H ALL DONE" >> artifacts/logs/j17h.log
