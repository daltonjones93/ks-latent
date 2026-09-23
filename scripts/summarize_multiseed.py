#!/usr/bin/env python
"""Summarize a multi-seed Stage-2 sweep (user-directed 2026-08-31): "run
like 10 random init markovian mlp stage 2 processes for both [the
fine-tuned AE] and [the original AE]. I want to see if all of them
converge in the second case, and all don't converge in the first case. or
if we just got lucky on the first run or something."

Parses `best_val_kmax_mse = <value>` out of each log matching --pattern
(named `stage2_multiseed_{group}_seed{N}.log` by the sweep script), groups
by `{group}` (the text between `multiseed_` and `_seed`), and reports
per-group min/median/mean/max plus a convergence count against
--threshold (default 0.05, comfortably above every well-converged run and
below every collapsed/stuck one seen in this document's own results).

    python scripts/summarize_multiseed.py --pattern "artifacts/logs/stage2_multiseed_*.log"
"""

from __future__ import annotations

import argparse
import glob
import re
import statistics


def _parse_group_seed(path: str) -> tuple[str, int]:
    m = re.search(r"stage2_multiseed_(.+)_seed(\d+)\.log$", path)
    if not m:
        raise ValueError(f"filename doesn't match the expected stage2_multiseed_{{group}}_seed{{N}}.log pattern: {path}")
    return m.group(1), int(m.group(2))


def _parse_best_val(path: str) -> float | None:
    text = open(path).read()
    m = re.search(r"best_val_kmax_mse = ([\d.eE+-]+)", text)
    return float(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--threshold", type=float, default=0.05,
                         help="best_val_kmax_mse at or below this counts as 'converged'.")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no files matched {args.pattern!r}")

    groups: dict[str, list[tuple[int, float | None]]] = {}
    for path in paths:
        group, seed = _parse_group_seed(path)
        val = _parse_best_val(path)
        groups.setdefault(group, []).append((seed, val))

    for group in sorted(groups):
        rows = sorted(groups[group])
        print(f"\n=== group: {group} ({len(rows)} seeds) ===")
        vals = []
        for seed, val in rows:
            status = f"{val:.6f}" if val is not None else "MISSING/INCOMPLETE"
            print(f"  seed {seed}: best_val_kmax_mse = {status}")
            if val is not None:
                vals.append(val)
        if vals:
            n_converged = sum(1 for v in vals if v <= args.threshold)
            print(f"  -- n={len(vals)}, min={min(vals):.6f}, median={statistics.median(vals):.6f}, "
                  f"mean={statistics.mean(vals):.6f}, max={max(vals):.6f}")
            print(f"  -- converged (<= {args.threshold}): {n_converged}/{len(vals)}")


if __name__ == "__main__":
    main()
