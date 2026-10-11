#!/bin/bash
# J17G driver: three levels; each starts its data once the previous level's is generated (trainings serialize inside).
cd "$(dirname "$0")/.."
scripts/run_j17g.sh 0.1 &
sleep 30; until [ -f artifacts/datasets/ks2d_gani0.1_wide6.h5 ]; do sleep 60; done
scripts/run_j17g.sh 0.2 &
sleep 30; until [ -f artifacts/datasets/ks2d_gani0.2_wide6.h5 ]; do sleep 60; done
scripts/run_j17g.sh 0.3 &
wait
echo "$(date '+%F %T') J17G ALL DONE" >> artifacts/logs/j17g.log
