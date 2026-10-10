#!/bin/bash
# I28 option 2, latent filter only (run early; run_i28c.sh lane a skips these once they exist).
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i28c.log
lat() { tag=$1; shift
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag p7c12_ks2d --n-tilings 4 --out $tag "$@" > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep -E "^$tag |DIVERGED" artifacts/logs/h9_${tag}.log | cut -c1-200 | tr '\n' ' ')" >> $LOG; }
for T in "adv1:--truth-advect-x 1.0" "adv2:--truth-advect-x 2.0" "frc0.2:--truth-forcing-amp 0.2" "frc0.4:--truth-forcing-amp 0.4"; do
  name=${T%%:*}; args=${T#*:}
  lat I28S_${name}_lat_rtps0.7 $args --rtps 0.7
  lat I28S_${name}_lat_rtps0.9_mn0.05 $args --rtps 0.9 --model-noise-std 0.05
done
echo "$(date '+%F %T') LANE latent-option2 DONE" >> $LOG
