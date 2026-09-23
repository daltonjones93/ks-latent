"""Tests for the §18 replication-result recording utility."""

from __future__ import annotations

import json

from ks_latent.utils.replication import record_result


def test_record_result_appends_jsonl(tmp_path, monkeypatch):
    from ks_latent.utils import replication

    path = tmp_path / "results.jsonl"
    monkeypatch.setattr(replication, "RESULTS_PATH", path)

    record_result("foo", 1.23, "[1, 2]", "must fall in range", True, units="1/time")
    record_result("bar", 4.0, ">= 3", "lower bound", False)

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0] == {
        "name": "foo", "value": 1.23, "target": "[1, 2]",
        "tolerance": "must fall in range", "passed": True, "units": "1/time",
    }
    assert lines[1]["name"] == "bar"
    assert lines[1]["passed"] is False
