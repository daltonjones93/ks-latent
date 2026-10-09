#!/bin/bash
# 2026-10-04: is the TRUE 2x/3x Kolmogorov flow (alpha=0.05) multistable? Long (2000 dt_snap = 400 time
# units) truth + P9-model rollouts from IC seeds 0,1,2; regime detection via the |k|<2 energy-share time series.
cd "$(dirname "$0")/.."
while pgrep -f "exp_rollout_stats_kolmogorov.py --tag p2_a005" > /dev/null || pgrep -f "exp_rollout_stats_kolmogorov.py --tag p9_a005 --with-truth$" > /dev/null; do sleep 30; done
for s in 0 1 2; do
  python scripts/exp_rollout_stats_kolmogorov.py --tag p9_a005 --scales 2,3 --seed $s --n-steps 2000 --with-truth \
    > artifacts/logs/multistab_seed$s.log 2>&1 && echo "[ms] seed $s done" || echo "[ms] seed $s FAILED"
done
