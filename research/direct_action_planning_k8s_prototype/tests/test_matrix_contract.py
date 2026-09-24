from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.run_matrix import matrix_rows, validate_matrix_config


def _config() -> dict:
    return {
        "schema": "dap.k8s.matrix_config.v1",
        "mode": "pilot",
        "source_split": "validation",
        "horizon_steps": 32,
        "control_interval_seconds": 10,
        "repetitions": 2,
        "seeds": [17, 23],
        "methods": ["static", "dap"],
        "budgets": {"low": 100, "high": 200},
        "profiles": {
            "azure_http": {"dataset": "azure2019", "domain": "http", "target_peak_rps": 300, "max_rps": 480}
        },
    }


def test_matrix_rows_pair_methods_on_same_profile_budget_seed() -> None:
    config = _config()
    validate_matrix_config(config)
    rows = matrix_rows(config)
    assert len(rows) == 8
    grouped = {}
    for row in rows:
        grouped.setdefault((row["profile"], row["budget_name"], row["seed"]), []).append(row["method"])
    assert all(sorted(methods) == ["dap", "static"] for methods in grouped.values())


def test_matrix_rejects_test_source_before_formal_freeze() -> None:
    config = _config()
    config["mode"] = "formal"
    config["source_split"] = "test"
    with pytest.raises(ValueError, match="frozen"):
        validate_matrix_config(config, require_frozen=True)


def test_matrix_run_id_is_stable_and_contains_no_timestamp() -> None:
    config = _config()
    rows = matrix_rows(config)
    assert rows[0]["run_id"] == "pilot__azure_http__low__seed17__static"
    assert all("2026" not in row["run_id"] for row in rows)
