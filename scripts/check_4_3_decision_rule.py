"""Phase E1 (`docs/steps_4-3.md`): mechanically apply Part 4.3's pre-
registered decision rule -- local-field + Gaspari-Cohn must beat the SEC
baseline on the GLOBAL latent, not merely beat no-localization.

Reads two CSVs produced by `scripts/run_da_ensemble_sweep.py`:
  - the GLOBAL-latent sweep (localizer in {none, sec})
  - the LOCAL-latent sweep (localizer in {none, gaspari_cohn})
and prints, for every `n_ensemble` present in BOTH csvs:
    rmse_gaspari_cohn[N] <= rmse_sec[N]  ->  PASS / FAIL

Per-N_ens results, not just an overall verdict -- a rule that only holds
at some ensemble sizes is itself informative (`docs/steps_4-3.md` E1).
Rows with `status == "FAILED"` (a run that itself crashed, e.g. a
genuinely singular ensemble covariance at very small N_ens) are reported
as `SKIPPED`, not silently dropped and not counted as PASS or FAIL.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _rmse_by_n_ensemble(rows: list[dict], localizer: str) -> dict[int, float | None]:
    """Returns {n_ensemble: rmse_da}, or None for a row whose own DA run
    failed (status == FAILED) -- kept in the dict (not dropped) so the
    caller can report SKIPPED rather than silently omitting that N_ens."""
    out: dict[int, float | None] = {}
    for row in rows:
        if row.get("localizer") != localizer:
            continue
        n_ens = int(float(row["n_ensemble"]))
        if row.get("status") == "FAILED" or not row.get("rmse_da"):
            out[n_ens] = None
        else:
            out[n_ens] = float(row["rmse_da"])
    return out


def apply_decision_rule(
    global_rows: list[dict], local_rows: list[dict],
) -> list[tuple[int, str, float | None, float | None]]:
    """Returns a list of (n_ensemble, verdict, rmse_gc, rmse_sec) tuples,
    verdict in {"PASS", "FAIL", "SKIPPED"}, sorted by n_ensemble. PASS
    means rmse_gaspari_cohn <= rmse_sec at that N_ens."""
    sec = _rmse_by_n_ensemble(global_rows, "sec")
    gc = _rmse_by_n_ensemble(local_rows, "gaspari_cohn")
    shared_n = sorted(set(sec.keys()) & set(gc.keys()))
    results = []
    for n_ens in shared_n:
        rmse_sec, rmse_gc = sec[n_ens], gc[n_ens]
        if rmse_sec is None or rmse_gc is None:
            results.append((n_ens, "SKIPPED", rmse_gc, rmse_sec))
        elif rmse_gc <= rmse_sec:
            results.append((n_ens, "PASS", rmse_gc, rmse_sec))
        else:
            results.append((n_ens, "FAIL", rmse_gc, rmse_sec))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-csv", required=True, help="Phase C sweep output (localizer sec present)")
    parser.add_argument("--local-csv", required=True, help="Phase D sweep output (localizer gaspari_cohn present)")
    args = parser.parse_args()

    global_rows = _load_rows(Path(args.global_csv))
    local_rows = _load_rows(Path(args.local_csv))
    results = apply_decision_rule(global_rows, local_rows)

    if not results:
        print("No shared n_ensemble values between the two CSVs -- nothing to check.")
        sys.exit(1)

    print(f"{'n_ensemble':>10}  {'verdict':>8}  {'rmse_gaspari_cohn':>18}  {'rmse_sec':>10}")
    n_pass = n_fail = n_skip = 0
    for n_ens, verdict, rmse_gc, rmse_sec in results:
        gc_str = f"{rmse_gc:.4f}" if rmse_gc is not None else "n/a"
        sec_str = f"{rmse_sec:.4f}" if rmse_sec is not None else "n/a"
        print(f"{n_ens:>10}  {verdict:>8}  {gc_str:>18}  {sec_str:>10}")
        n_pass += verdict == "PASS"
        n_fail += verdict == "FAIL"
        n_skip += verdict == "SKIPPED"
    print(f"\n{n_pass} PASS, {n_fail} FAIL, {n_skip} SKIPPED (of {len(results)} shared n_ensemble values)")


if __name__ == "__main__":
    main()
