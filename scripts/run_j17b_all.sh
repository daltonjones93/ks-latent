#!/bin/bash
# J17B driver: both pre-registered levels; the second starts its data once the first's is generated.
cd "$(dirname "$0")/.."
scripts/run_j17b.sh 0.1 &
sleep 30; until [ -f artifacts/datasets/ks2d_memB0.1_wide6.h5 ]; do sleep 60; done
scripts/run_j17b.sh 0.2 &
wait
echo "$(date '+%F %T') J17B ALL DONE" >> artifacts/logs/j17b.log
