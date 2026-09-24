from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.aggregate import aggregate_run


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_run_uses_one_complete_run_as_unit(tmp_path: Path) -> None:
    run = tmp_path / "pilot__azure_http__low__seed17__dap"
    run.mkdir()
    _write_json(run / "run_manifest.json", {"status": "completed", "method": "dap", "profile": "azure_http", "budget_seconds": 100.0, "matrix_row_id": "x"})
    _write_json(run / "result.json", {
        "status": "completed", "method": "dap", "profile": "azure_http", "budget_seconds": 100.0,
        "replay": {"scheduled": 4, "sent": 4, "completed": 3, "failed": 1, "timed_out": 1, "wall_seconds": 2.0},
        "controller": {"ready_cost_seconds": 30.0, "requested_cost_seconds": 32.0, "remaining_budget_seconds": 70.0, "budget_violation_seconds": 0.0, "deadline_misses": 0},
    })
    (run / "requests.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"client_latency_seconds": 0.10, "http_status": 200, "error": None},
        {"client_latency_seconds": 0.20, "http_status": 200, "error": None},
        {"client_latency_seconds": 0.30, "http_status": 200, "error": None},
        {"client_latency_seconds": 2.00, "http_status": 504, "error": "timeout"},
    ]) + "\n", encoding="utf-8")
    (run / "controller").mkdir()
    (run / "controller" / "controller_actions.jsonl").write_text(
        json.dumps({"target_replicas": 1}) + "\n" + json.dumps({"target_replicas": 3}) + "\n",
        encoding="utf-8",
    )
    row = aggregate_run(run, slo_seconds=0.25)
    assert row["run_status"] == "completed"
    assert row["completed_requests"] == 3
    assert row["slo_violation_rate"] == pytest.approx(0.5)
    assert row["ready_replica_seconds"] == pytest.approx(30.0)
    assert row["scaling_count"] == 1
