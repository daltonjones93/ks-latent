"""Tests for `scripts/check_4_3_decision_rule.py` (Phase E1,
`docs/steps_4-3.md`): the decision-script itself gets a unit test before
being trusted on real sweep data, per that phase's own explicit
requirement.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_4_3_decision_rule import apply_decision_rule  # noqa: E402


def _rows(*dicts):
    return list(dicts)


def test_local_wins_at_every_n_ensemble_reports_all_pass():
    global_rows = _rows(
        {"n_ensemble": "8", "localizer": "sec", "rmse_da": "1.0", "status": "OK"},
        {"n_ensemble": "16", "localizer": "sec", "rmse_da": "0.9", "status": "OK"},
        {"n_ensemble": "32", "localizer": "sec", "rmse_da": "0.8", "status": "OK"},
    )
    local_rows = _rows(
        {"n_ensemble": "8", "localizer": "gaspari_cohn", "rmse_da": "0.5", "status": "OK"},
        {"n_ensemble": "16", "localizer": "gaspari_cohn", "rmse_da": "0.5", "status": "OK"},
        {"n_ensemble": "32", "localizer": "gaspari_cohn", "rmse_da": "0.5", "status": "OK"},
    )
    results = apply_decision_rule(global_rows, local_rows)
    assert [r[1] for r in results] == ["PASS", "PASS", "PASS"]


def test_local_loses_at_every_n_ensemble_reports_all_fail():
    global_rows = _rows(
        {"n_ensemble": "8", "localizer": "sec", "rmse_da": "0.3", "status": "OK"},
        {"n_ensemble": "16", "localizer": "sec", "rmse_da": "0.3", "status": "OK"},
    )
    local_rows = _rows(
        {"n_ensemble": "8", "localizer": "gaspari_cohn", "rmse_da": "0.9", "status": "OK"},
        {"n_ensemble": "16", "localizer": "gaspari_cohn", "rmse_da": "0.9", "status": "OK"},
    )
    results = apply_decision_rule(global_rows, local_rows)
    assert [r[1] for r in results] == ["FAIL", "FAIL"]


def test_mixed_result_reports_per_n_ensemble_not_just_overall():
    """A rule that only holds at SOME ensemble sizes must be visible per
    N_ens, not collapsed into one verdict -- this is the whole point of
    the per-row table, not an edge case."""
    global_rows = _rows(
        {"n_ensemble": "8", "localizer": "sec", "rmse_da": "1.0", "status": "OK"},
        {"n_ensemble": "64", "localizer": "sec", "rmse_da": "0.3", "status": "OK"},
    )
    local_rows = _rows(
        {"n_ensemble": "8", "localizer": "gaspari_cohn", "rmse_da": "0.5", "status": "OK"},
        {"n_ensemble": "64", "localizer": "gaspari_cohn", "rmse_da": "0.6", "status": "OK"},
    )
    results = apply_decision_rule(global_rows, local_rows)
    by_n = {r[0]: r[1] for r in results}
    assert by_n[8] == "PASS"
    assert by_n[64] == "FAIL"


def test_exact_tie_counts_as_pass():
    """rule is <=, not <: an exact tie should not count as a failure."""
    global_rows = _rows({"n_ensemble": "32", "localizer": "sec", "rmse_da": "0.5", "status": "OK"})
    local_rows = _rows({"n_ensemble": "32", "localizer": "gaspari_cohn", "rmse_da": "0.5", "status": "OK"})
    results = apply_decision_rule(global_rows, local_rows)
    assert results[0][1] == "PASS"


def test_failed_run_is_skipped_not_silently_dropped_or_counted():
    global_rows = _rows({"n_ensemble": "8", "localizer": "sec", "rmse_da": "1.0", "status": "OK"})
    local_rows = _rows({"n_ensemble": "8", "localizer": "gaspari_cohn", "status": "FAILED", "error": "boom"})
    results = apply_decision_rule(global_rows, local_rows)
    assert len(results) == 1
    assert results[0][1] == "SKIPPED"


def test_only_shared_n_ensemble_values_are_compared():
    global_rows = _rows(
        {"n_ensemble": "8", "localizer": "sec", "rmse_da": "1.0", "status": "OK"},
        {"n_ensemble": "999", "localizer": "sec", "rmse_da": "1.0", "status": "OK"},
    )
    local_rows = _rows(
        {"n_ensemble": "8", "localizer": "gaspari_cohn", "rmse_da": "0.5", "status": "OK"},
    )
    results = apply_decision_rule(global_rows, local_rows)
    assert len(results) == 1
    assert results[0][0] == 8


def test_ignores_none_localizer_rows_uses_only_sec_and_gaspari_cohn():
    global_rows = _rows(
        {"n_ensemble": "8", "localizer": "none", "rmse_da": "5.0", "status": "OK"},
        {"n_ensemble": "8", "localizer": "sec", "rmse_da": "1.0", "status": "OK"},
    )
    local_rows = _rows(
        {"n_ensemble": "8", "localizer": "none", "rmse_da": "5.0", "status": "OK"},
        {"n_ensemble": "8", "localizer": "gaspari_cohn", "rmse_da": "0.9", "status": "OK"},
    )
    results = apply_decision_rule(global_rows, local_rows)
    assert len(results) == 1
    assert results[0][1] == "PASS"  # 0.9 <= 1.0, using sec/gaspari_cohn rows only, not the 'none' rows


def test_csv_round_trip_smoke(tmp_path):
    """Sanity check against actual CSV files (as scripts/run_da_ensemble_
    sweep.py would produce), not just in-memory dicts."""
    global_csv = tmp_path / "global.csv"
    local_csv = tmp_path / "local.csv"
    with global_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_ensemble", "localizer", "rmse_da", "status"])
        w.writeheader()
        w.writerow({"n_ensemble": "16", "localizer": "sec", "rmse_da": "0.7", "status": "OK"})
    with local_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_ensemble", "localizer", "rmse_da", "status"])
        w.writeheader()
        w.writerow({"n_ensemble": "16", "localizer": "gaspari_cohn", "rmse_da": "0.6", "status": "OK"})

    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "check_4_3_decision_rule.py"),
         "--global-csv", str(global_csv), "--local-csv", str(local_csv)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "PASS" in result.stdout
    assert "1 PASS, 0 FAIL, 0 SKIPPED" in result.stdout
