#!/bin/bash
# Driver for scripts/run_j17g_tune.sh: delta = 0.2 and 0.1 side by side (3 lanes each), then 0.3.
cd "$(dirname "$0")/.."
scripts/run_j17g_tune.sh 0.2 3 & scripts/run_j17g_tune.sh 0.1 3 & wait
scripts/run_j17g_tune.sh 0.3 4
echo "$(date '+%F %T') J17G TUNE ALL DONE" >> artifacts/logs/j17g_tune.log
