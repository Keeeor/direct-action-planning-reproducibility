from __future__ import annotations

import pytest

from dap.direct_action_planning_k8s_pareto_calibration.analysis import (
    action_horizon_valid,
    paired_candidate_rows,
    summarize_screen_candidates,
)


def _row(
    method: str,
    seed: int,
    *,
    completion: float,
    slo: float,
    cost: float,
    budget_violation: float = 0.0,
) -> dict[str, float | int | str]:
    return {
        "method": method,
        "seed": seed,
        "completion_rate": completion,
        "slo_violation_rate": slo,
        "ready_replica_seconds": cost,
        "budget_violation_seconds": budget_violation,
    }


def test_threshold_action_horizon_uses_log_when_optional_steps_absent() -> None:
    actions = [{"step": step} for step in range(32)]
    assert action_horizon_valid(actions, {}) is True
    assert action_horizon_valid(actions, {"steps": 32}) is True
    assert action_horizon_valid(actions, {"steps": 31}) is False
    assert action_horizon_valid(actions[:-1], {}) is False


def test_paired_rows_use_candidate_minus_threshold_endpoint_directions() -> None:
    rows = [
        _row("threshold", 1, completion=0.90, slo=0.20, cost=100.0),
        _row("dap_cont_0p10", 1, completion=0.92, slo=0.17, cost=108.0),
    ]
    paired = paired_candidate_rows(rows, candidate_methods=("dap_cont_0p10",))
    assert len(paired) == 1
    assert paired[0]["method"] == "dap_cont_0p10"
    assert paired[0]["seed"] == 1
    assert paired[0]["completion_gain"] == pytest.approx(0.02)
    assert paired[0]["completion_loss"] == pytest.approx(-0.02)
    assert paired[0]["slo_increase"] == pytest.approx(-0.03)
    assert paired[0]["ready_cost_increase_seconds"] == pytest.approx(8.0)
    assert paired[0]["budget_violation_seconds"] == 0.0


def test_screen_summary_uses_means_and_worst_budget_violation() -> None:
    paired = [
        {
            "method": "dap_cont_0p10",
            "seed": 1,
            "completion_gain": 0.02,
            "completion_loss": -0.02,
            "slo_increase": -0.03,
            "ready_cost_increase_seconds": 8.0,
            "budget_violation_seconds": 0.0,
        },
        {
            "method": "dap_cont_0p10",
            "seed": 2,
            "completion_gain": 0.00,
            "completion_loss": 0.00,
            "slo_increase": 0.01,
            "ready_cost_increase_seconds": 12.0,
            "budget_violation_seconds": 1.0e-10,
        },
    ]
    summary = summarize_screen_candidates(
        paired,
        continuation_candidates={"dap_cont_0p10": 0.10},
    )
    assert len(summary) == 1
    row = summary[0]
    assert row["n_pairs"] == 2
    assert abs(row["completion_gain"] - 0.01) < 1.0e-12
    assert abs(row["completion_loss"] + 0.01) < 1.0e-12
    assert abs(row["slo_increase"] + 0.01) < 1.0e-12
    assert abs(row["ready_cost_increase_seconds"] - 10.0) < 1.0e-12
    assert row["budget_violation_seconds"] == 1.0e-10
