#!/bin/bash
# Waits for the IC-normalization check to finish, then runs C0, then C4 -- strictly sequential.
while pgrep -f run_ic_normalization_check.sh > /dev/null; do sleep 30; done
echo "[$(date +%T)] icnorm finished; starting C0"
python scripts/exp_c0_capacity_audit.py > artifacts/logs/c0_runner.log 2>&1 && echo "[$(date +%T)] C0 DONE" || echo "[$(date +%T)] C0 FAILED"
echo "[$(date +%T)] starting C4"
python scripts/exp_c4_reprojection.py > artifacts/logs/c4_runner.log 2>&1 && echo "[$(date +%T)] C4 DONE" || echo "[$(date +%T)] C4 FAILED"
