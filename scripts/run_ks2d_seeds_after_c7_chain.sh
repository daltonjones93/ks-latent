#!/bin/bash
# Reordered 2026-10-03 (user: "should we just move to C7?"):
# wait for the in-flight P2 eval -> C7 -> crop pipeline remainder (alpha=0.05 3x truth)
# -> 2D-KS normalized-IC seeds. (Name contains "run_ks2d_seeds_after" so the queued T1
# x2-truth waiter and the scheduling guard keep waiting on this chain.)
cd "$(dirname "$0")/.."
while pgrep -f exp_crop_eval.py > /dev/null; do sleep 20; done
python scripts/exp_c7_pipeline.py >> artifacts/logs/c7_pipeline.log 2>&1
python scripts/exp_crop_pipeline.py >> artifacts/logs/crop_pipeline.log 2>&1
bash scripts/run_icnorm_ks2d_seeds.sh >> artifacts/logs/icnorm_ks2d_seeds.log 2>&1
echo "[$(date +%T)] chain done"
