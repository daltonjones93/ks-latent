#!/bin/bash
# Re-applies normal (non-background) scheduling to this project's experiment
# processes every minute; macOS otherwise demotes them to efficiency cores.
# Exits when no experiment pipeline is left running.
PAT="exp_crop_pipeline|exp_crop_eval|exp_c7_pipeline|train_kolmogorov_crop|gen_kolmogorov_wide|exp_ic_normalization_check|run_ks2d_seeds_after|run_icnorm_ks2d_seeds"
while pgrep -f "exp_crop_pipeline|exp_c7_pipeline|run_ks2d_seeds_after|run_icnorm_ks2d_seeds" > /dev/null; do
  for p in $(pgrep -f "$PAT"); do taskpolicy -B -t 0 -l 0 -p $p 2>/dev/null; done
  sleep 60
done
