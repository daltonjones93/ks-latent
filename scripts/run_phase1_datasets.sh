#!/bin/bash
# Consolidation Phase 1 data: 6x6-box Kolmogorov datasets at alpha=0.1 and 0.025 (alpha=0.05 exists). Normal priority.
cd "$(dirname "$0")/.."
for spec in "0.1 910" "0.025 920"; do set -- $spec
  out=artifacts/datasets/kolmogorov_wide6x6_alpha$1.h5
  [ -f $out ] || python scripts/gen_kolmogorov_wide_dataset.py --alpha $1 --x-scale 6 --y-scale 6 --seed-base $2 \
    --out $out >> artifacts/logs/phase1_data_a$1.log 2>&1 &
done; wait; echo "[p1data] done"
