"""E2 (docs/ks_local_field_ltransfer.tex): full N_ens sweep at the
L-transfer sizes, with and without localization -- turns Phase F3's
single-point result (docs/RESULTS.md: N_ens=64, L=200, one gc_c) into a
scaling curve.

Loops `scripts/run_da_pff_ltransfer.py` (Phase F3's own DA-cycling driver,
which loads Section 224's checkpoint resized to the target L with ZERO
retraining) over `--n-ensemble` x `--localizer`, collecting each run's
JSON into one CSV -- same thin-wrapper-over-subprocess convention as
`scripts/run_da_ensemble_sweep.py` uses for the L=100 sweep, deliberately
not reimplementing any DA logic here.

Predicted, falsifiable signature (docs/ks_local_field_ltransfer.tex,
Section E2): the smallest N_ens clearing a fixed skill threshold should
stay roughly flat in L for the LOCALIZED curve (tracks local per-site
dimension, held fixed by the transfer) and grow with L for the
UNLOCALIZED curve (tracks the doubling/quadrupling global dimension).
This script only collects the raw sweep; `--skill-threshold` just
annotates each row with whether it clears the bar so that threshold-
crossing point is easy to read off the CSV directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--prop-checkpoint", required=True)
    parser.add_argument("--old-L", type=float, default=100.0)
    parser.add_argument("--new-L", type=float, required=True)
    parser.add_argument("--n-prop-steps", type=int, default=5)
    parser.add_argument("--n-cycles", type=int, default=40)
    parser.add_argument("--obs-stride-at-old-L", type=int, default=8)
    parser.add_argument("--dt-snap", type=float, default=1.0)
    parser.add_argument("--n-ensemble-list", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--localizers", nargs="+", default=["none", "gaspari_cohn"], choices=["none", "gaspari_cohn"])
    parser.add_argument("--gc-c", type=float, default=3.0, help="In SITE units at the NEW (transferred) n_sites -- same value used at L=100/L=200 (docs/RESULTS.md Phase D2/F3).")
    parser.add_argument(
        "--method", choices=["pff", "letkf"], default="pff",
        help="DA analysis scheme (added 2026-09-28 for E7): 'pff' (default, preserves E2's original "
        "behavior/tags exactly) or 'letkf' (ks_latent.da.letkf, for the E7 sweep).",
    )
    parser.add_argument("--letkf-inflation", type=float, default=1.0)
    parser.add_argument("--obs-spacing-units", type=float, default=3.125,
                        help="Fixed physical observation spacing (task D1); ignored with --legacy-scaled-stride.")
    parser.add_argument("--legacy-scaled-stride", action="store_true",
                        help="Reproduce the original E2/E7 sweeps (stride scaled with L).")
    parser.add_argument("--skill-threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    rows = []
    for localizer in args.localizers:
        for n_ens in args.n_ensemble_list:
            tag = f"{args.tag_prefix}_{localizer}_nens{n_ens}"
            cmd = [
                sys.executable, str(REPO_ROOT / "scripts" / "run_da_pff_ltransfer.py"),
                "--ae-checkpoint", args.ae_checkpoint,
                "--prop-checkpoint", args.prop_checkpoint,
                "--old-L", str(args.old_L), "--new-L", str(args.new_L),
                "--n-ensemble", str(n_ens),
                "--n-prop-steps", str(args.n_prop_steps),
                "--n-cycles", str(args.n_cycles),
                "--obs-stride-at-old-L", str(args.obs_stride_at_old_L),
                "--dt-snap", str(args.dt_snap),
                "--seed", str(args.seed),
                "--localizer", localizer,
                "--method", args.method,
                "--letkf-inflation", str(args.letkf_inflation),
                "--obs-spacing-units", str(args.obs_spacing_units),
                "--tag", tag,
            ]
            if args.legacy_scaled_stride:
                cmd += ["--legacy-scaled-stride"]
            if localizer == "gaspari_cohn":
                cmd += ["--gc-c", str(args.gc_c)]
            print(f"=== new_L={args.new_L} localizer={localizer} n_ensemble={n_ens} ===", flush=True)
            result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                last_error_line = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "(no stderr)"
                print(f"  FAILED (exit {result.returncode}): {last_error_line}")
                rows.append({
                    "new_L": args.new_L, "n_ensemble": n_ens, "localizer": localizer,
                    "gc_c": args.gc_c if localizer == "gaspari_cohn" else None,
                    "status": "FAILED", "error": last_error_line,
                })
                continue
            report_path = REPO_ROOT / "artifacts" / f"da_pff_ltransfer_{tag}.json"
            report = json.loads(report_path.read_text())
            report["status"] = "OK"
            report["clears_skill_threshold"] = report["skill_free_over_da"] >= args.skill_threshold
            print(json.dumps({k: v for k, v in report.items() if not k.endswith("_per_step")}, indent=2))
            rows.append({k: v for k, v in report.items() if not k.endswith("_per_step")})  # arrays stay in the JSON

    out_path = REPO_ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")

    print("\n=== smallest N_ens clearing skill >= %.2f, per localizer ===" % args.skill_threshold)
    for localizer in args.localizers:
        ok = [r for r in rows if r.get("localizer") == localizer and r.get("status") == "OK" and r.get("clears_skill_threshold")]
        if ok:
            best = min(int(float(r["n_ensemble"])) for r in ok)
            print(f"  {localizer}: N_ens={best}")
        else:
            print(f"  {localizer}: none of the tested N_ens cleared the threshold")


if __name__ == "__main__":
    main()
