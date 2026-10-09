"""E3 sweep driver (docs/ks_local_field_ltransfer.tex): loops
`scripts/exp_e3_sec_vs_gc_at_transfer.py` (SEC vs. Gaspari-Cohn, BOTH
applied to the SAME resized local latent -- see that script's own
docstring for why this is a cleaner ablation than the original L=100
SEC-vs-GC comparison, which had a global-vs-local architecture confound)
over `--n-ensemble` x `--localizer`, collecting one CSV -- same
thin-wrapper-over-subprocess convention as
`scripts/run_da_ensemble_sweep.py`/`scripts/exp_e2_ensemble_sweep_ltransfer.py`.

The resulting CSV is directly consumable by
`scripts/check_4_3_decision_rule.py --global-csv <this> --local-csv <this>`
(same CSV passed for both arguments is valid: SEC and Gaspari-Cohn rows
both live in one file here, unlike the original L=100 comparison which
needed two separate checkpoints/CSVs).
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
    parser.add_argument("--localizers", nargs="+", default=["sec", "gaspari_cohn"], choices=["none", "sec", "gaspari_cohn"])
    parser.add_argument("--gc-c", type=float, default=3.0)
    parser.add_argument("--sec-n-trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    rows = []
    for localizer in args.localizers:
        for n_ens in args.n_ensemble_list:
            tag = f"{args.tag_prefix}_{localizer}_nens{n_ens}"
            cmd = [
                sys.executable, str(REPO_ROOT / "scripts" / "exp_e3_sec_vs_gc_at_transfer.py"),
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
                "--sec-n-trials", str(args.sec_n_trials),
                "--tag", tag,
            ]
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
            report_path = REPO_ROOT / "artifacts" / f"e3_sec_vs_gc_ltransfer_{tag}.json"
            report = json.loads(report_path.read_text())
            report["status"] = "OK"
            rows.append(report)
            print(json.dumps(report, indent=2))

    out_path = REPO_ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
