"""Drift test for a DA run (Block I protocol, 2026-10-09): compare the blended analysis error over
the two quarters of the second half of the run; 'steady' if they differ by < 10%.
Usage: python scripts/drift_check.py h9_<tag> [...]"""
import json
import sys

import numpy as np

for t in sys.argv[1:]:
    d = json.load(open(f"artifacts/experiments/da2d/{t}.json"))
    for key in ("blend", "tiling0"):
        v = np.array(d["per_cycle"][key])
        n = len(v); a, b = v[n // 2: 3 * n // 4].mean(), v[3 * n // 4:].mean()
        print(f"{t:40s} {key:8s} 2nd-half Q1 {a:.3f} Q2 {b:.3f} change {100 * (b - a) / a:+.0f}% "
              f"{'steady' if abs(b - a) / a < 0.10 else 'DRIFTING'}")
