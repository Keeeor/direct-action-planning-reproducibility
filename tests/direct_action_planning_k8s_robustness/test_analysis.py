from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage2_dynamic_budget.direct_action_planning_k8s_robustness.analysis import (
    audit_matrix,
    pair_with_historical,
    select_completed_runs,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(
    root: Path,
    *,
    name: str,
    contract: str,
    profile: str,
    condition: str,
    seed: int,
    status: str = "completed",
    applied: int = 6,
) -> Path:
    run = root / name
    write_json(
        run / "run_manifest.json",
        {
            "status": status,
            "robustness_contract_sha256": contract,
            "profile": profile,
            "perturbation_condition": condition,
            "seed": seed,
            "plan_split": "test",
        },
    )
    write_json(
        run / "perturbation_delivery.json",
        {"status": "PASS", "applied_events": applied, "registered_events": applied},
    )
    write_json(
        run / "readiness_restoration.json",
        {"status": "PASS", "restored_initial_delay_seconds": 1},
    )
    write_json(
        run / "result.json",
        {
            "status": "completed",
            "controller": {
                "budget_violation_seconds": 0.0,
                "deadline_misses": 0,
            },
            "monitor": {"failures": 0},
        },
    )
    return run


def test_select_completed_runs_ignores_old_contracts_and_failed_attempts(tmp_path):
    wanted = "sha256:v3"
    make_run(
        tmp_path,
        name="old",
        contract="sha256:v2",
        profile="azure_http",
        condition="metric_dropout_10pct",
        seed=1,
    )
    make_run(
        tmp_path,
        name="failed",
        contract=wanted,
        profile="azure_http",
        condition="metric_dropout_10pct",
        seed=1,
        status="failed",
    )
    expected = make_run(
        tmp_path,
        name="complete",
        contract=wanted,
        profile="azure_http",
        condition="metric_dropout_10pct",
        seed=1,
    )
    assert select_completed_runs(tmp_path, wanted) == [expected]


def test_audit_matrix_requires_exact_cells_delivery_budget_and_restoration(tmp_path):
    contract = "sha256:v3"
    profiles = ["azure_http", "gentd_inference"]
    conditions = ["metric_dropout_10pct"]
    seeds = [1, 2]
    runs = []
    for profile in profiles:
        for seed in seeds:
            runs.append(
                make_run(
                    tmp_path,
                    name=f"{profile}-{seed}",
                    contract=contract,
                    profile=profile,
                    condition=conditions[0],
                    seed=seed,
                )
            )
    config = {
        "profiles": {name: {} for name in profiles},
        "conditions": {name: {} for name in conditions},
        "seeds": seeds,
    }
    audit = audit_matrix(runs, config, contract)
    assert audit["passed"] is True
    assert audit["registered_cells"] == 4
    assert audit["completed_cells"] == 4
    assert audit["total_applied_events"] == 24
    assert audit["max_budget_violation_seconds"] == 0.0

    broken = json.loads((runs[0] / "result.json").read_text())
    broken["controller"]["budget_violation_seconds"] = 0.25
    write_json(runs[0] / "result.json", broken)
    failed_audit = audit_matrix(runs, config, contract)
    assert failed_audit["passed"] is False
    assert failed_audit["max_budget_violation_seconds"] == 0.25


def test_pair_with_historical_uses_profile_seed_plan_and_explicit_delta_direction():
    perturbed = pd.DataFrame(
        [
            {
                "profile": "azure_http",
                "seed": 1,
                "condition": "observation_lag_1",
                "plan_sha256": "same",
                "completion_rate": 0.90,
                "ready_replica_seconds": 100.0,
            }
        ]
    )
    historical = pd.DataFrame(
        [
            {
                "profile": "azure_http",
                "seed": 1,
                "plan_sha256": "same",
                "completion_rate": 0.95,
                "ready_replica_seconds": 110.0,
            }
        ]
    )
    paired = pair_with_historical(
        perturbed,
        historical,
        metrics=("completion_rate", "ready_replica_seconds"),
    )
    assert paired.loc[0, "delta_definition"] == "perturbed_minus_historical"
    assert paired.loc[0, "completion_rate_delta"] == pytest.approx(-0.05)
    assert paired.loc[0, "ready_replica_seconds_delta"] == -10.0

    historical.loc[0, "plan_sha256"] = "different"
    try:
        pair_with_historical(perturbed, historical, metrics=("completion_rate",))
    except ValueError as error:
        assert "plan hash" in str(error)
    else:
        raise AssertionError("plan mismatch must be rejected")
