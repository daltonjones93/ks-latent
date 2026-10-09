#!/bin/bash
# Overnight (2026-10-04, user asleep): keep every experiment process on background QoS (efficiency cores).
# Re-applies taskpolicy -b every 60 s to any matching process (new queue steps included). Kill to stop.
while true; do
  for p in $(pgrep -f "run_p9b_c19_p8|run_multistability|run_p7_x2|exp_rollout_stats|train_kolmogorov|exp_crop_eval|run_truth_kolmogorov|gen_kolmogorov"); do
    taskpolicy -b -p $p 2>/dev/null
  done
  sleep 60
done
