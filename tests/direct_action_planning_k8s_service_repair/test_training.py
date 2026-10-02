from __future__ import annotations

import numpy as np

from dap.direct_action_planning_k8s_service_repair.training import (
    select_validation_candidate,
)


def test_validation_selection_enforces_service_guard_before_productivity() -> None:
    comparator = {
        "completion_ratio": 0.90,
        "slo_violation_rate": 0.20,
        "ready_cost_seconds": 100.0,
    }
    candidates = [
        {
            "iteration": 2, "continuation_weight": 0.25, "tie_margin": 0.0,
            "completion_ratio": 0.70, "slo_violation_rate": 0.10,
            "ready_cost_seconds": 10.0, "target_changes": 1,
            "budget_violation_seconds": 0.0,
        },
        {
            "iteration": 4, "continuation_weight": 0.5, "tie_margin": 0.03,
            "completion_ratio": 0.90, "slo_violation_rate": 0.20,
            "ready_cost_seconds": 80.0, "target_changes": 3,
            "budget_violation_seconds": 0.0,
        },
    ]
    selected = select_validation_candidate(candidates, comparator)
    assert selected["iteration"] == 4
    assert selected["passed_service_guard"]


def test_validation_selection_is_deterministic_on_full_tie() -> None:
    comparator = {
        "completion_ratio": 0.9, "slo_violation_rate": 0.2,
        "ready_cost_seconds": 100.0,
    }
    rows = [
        {
            "iteration": iteration, "continuation_weight": weight,
            "tie_margin": margin, "completion_ratio": 0.9,
            "slo_violation_rate": 0.2, "ready_cost_seconds": 80.0,
            "target_changes": 2, "budget_violation_seconds": 0.0,
        }
        for iteration, weight, margin in ((4, 0.5, 0.03), (2, 0.25, 0.0))
    ]
    assert select_validation_candidate(rows, comparator)["iteration"] == 2


def test_validation_selection_retains_least_violating_diagnostic_if_no_pass() -> None:
    comparator = {
        "completion_ratio": 0.9, "slo_violation_rate": 0.2,
        "ready_cost_seconds": 100.0,
    }
    rows = [
        {
            "iteration": 2, "continuation_weight": 0.25, "tie_margin": 0.0,
            "completion_ratio": 0.88, "slo_violation_rate": 0.21,
            "ready_cost_seconds": 50.0, "target_changes": 2,
            "budget_violation_seconds": 0.0,
        },
        {
            "iteration": 4, "continuation_weight": 0.5, "tie_margin": 0.0,
            "completion_ratio": 0.80, "slo_violation_rate": 0.30,
            "ready_cost_seconds": 20.0, "target_changes": 1,
            "budget_violation_seconds": 0.0,
        },
    ]
    selected = select_validation_candidate(rows, comparator)
    assert selected["iteration"] == 2
    assert not selected["passed_service_guard"]
    assert np.isclose(selected["guard_violation"], 0.01)

