"""Phase C1/D3 sweep driver (`docs/steps_4-3.md`, added 2026-09-25).

Loops `scripts/run_da_pff.py` over `--n-ensemble` x `--localizer`
(and, for `gaspari_cohn`, a fixed `--gc-c`) on one checkpoint, collecting
each run's JSON output into a single combined table -- a thin wrapper,
not a fancier aggregator, per `docs/steps_4-3.md` Phase C1's own note
that nothing fancier is needed here.

Each individual `run_da_pff.py` call is a subprocess (not an import) --
matches this project's own convention of treating `run_da_pff.py` as
the single source of truth for the actual DA cycling logic; this script
only drives it and reads back its JSON, never reimplements any of it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--prop-checkpoint", required=True)
    parser.add_argument("--n-prop-steps", type=int, required=True)
    parser.add_argument("--dt-snap", type=float, default=1.0)
    parser.add_argument("--L", type=float, default=100.0)
    parser.add_argument("--n-cycles", type=int, default=40)
    parser.add_argument(
        "--n-ensemble-list", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--localizers", nargs="+", default=["none", "sec"], choices=["none", "sec", "gaspari_cohn"],
    )
    parser.add_argument("--gc-c", type=float, default=2.0, help="--localizer gaspari_cohn only.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    rows = []
    for localizer in args.localizers:
        for n_ens in args.n_ensemble_list:
            tag = f"{args.tag_prefix}_{localizer}_nens{n_ens}"
            cmd = [
                sys.executable, str(REPO_ROOT / "scripts" / "run_da_pff.py"),
                "--profile", "full",
                "--ae-checkpoint", args.ae_checkpoint,
                "--prop-checkpoint", args.prop_checkpoint,
                "--n-ensemble", str(n_ens),
                "--n-prop-steps", str(args.n_prop_steps),
                "--n-cycles", str(args.n_cycles),
                "--dt-snap", str(args.dt_snap),
                "--L", str(args.L),
                "--seed", str(args.seed),
                "--localizer", localizer,
                "--tag", tag,
            ]
            if localizer == "gaspari_cohn":
                cmd += ["--gc-c", str(args.gc_c)]
            print(f"=== localizer={localizer} n_ensemble={n_ens} ===", flush=True)
            result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                # A single combination failing (e.g. a genuinely rank-
                # deficient ensemble covariance at very small n_ensemble
                # relative to d_latent) is itself an informative result,
                # not a harness bug -- record it loudly (last error line,
                # not swallowed) and keep sweeping the rest rather than
                # aborting the whole table over one point.
                last_error_line = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "(no stderr)"
                print(f"  FAILED (exit {result.returncode}): {last_error_line}")
                rows.append({
                    "n_ensemble": n_ens, "localizer": localizer, "gc_c": args.gc_c if localizer == "gaspari_cohn" else None,
                    "status": "FAILED", "error": last_error_line,
                })
                continue
            report_path = REPO_ROOT / "artifacts" / f"da_pff_full_{tag}.json"
            report = json.loads(report_path.read_text())
            report["status"] = "OK"
            rows.append(report)
            print(json.dumps(report, indent=2))

    out_path = REPO_ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    import csv
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
