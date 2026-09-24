from __future__ import annotations

from stage2_dynamic_budget.direct_action_planning_k8s_cost_calibration.selection import (
    SelectionGuards,
    select_cost_aware_candidate,
)


def _metrics(completion: float, slo: float, cost: float, changes: float = 5.0):
    return {
        "completion_ratio": completion,
        "slo_violation_rate": slo,
        "ready_cost_seconds": cost,
        "target_changes": changes,
        "budget_violation_seconds": 0.0,
    }


def _candidate(*, cost: float, low_cost: float, high_completion: float = 0.96,
               high_slo: float = 0.08, cost_weight: float = 0.2,
               continuation: float = 0.1, iteration: int = 4,
               tie_margin: float = 0.01):
    return {
        "cost_weight": cost_weight,
        "continuation_weight": continuation,
        "iteration": iteration,
        "tie_margin": tie_margin,
        "global": _metrics(0.95, 0.10, cost),
        "low_activity": _metrics(1.0, 0.0, low_cost),
        "high_activity": _metrics(high_completion, high_slo, cost + 30.0),
    }


def _comparator():
    return {
        "global": _metrics(0.95, 0.10, 100.0),
        "low_activity": _metrics(1.0, 0.0, 2.0),
        "high_activity": _metrics(0.95, 0.09, 130.0),
    }


def test_rejects_service_safe_candidate_with_low_activity_waste() -> None:
    wasteful = _candidate(cost=105.0, low_cost=60.0)
    safe = _candidate(cost=104.0, low_cost=18.0, continuation=0.0)
    selected = select_cost_aware_candidate(
        [wasteful, safe], _comparator(), SelectionGuards()
    )
    assert selected["continuation_weight"] == 0.0
    assert selected["passed_all_guards"] is True
    rejected = selected["selection_diagnostics"]["evaluated_candidates"][0]
    assert rejected["passed_low_activity_cost_guard"] is False


def test_global_relative_or_absolute_cost_guard_is_enforced() -> None:
    # Threshold cost is 100, so the allowed increase is max(10, 5)=10 seconds.
    expensive = _candidate(cost=111.0, low_cost=10.0)
    selected = select_cost_aware_candidate(
        [expensive], _comparator(), SelectionGuards()
    )
    assert selected["passed_all_guards"] is False
    assert selected["passed_global_cost_guard"] is False
    assert selected["selection_status"] == "diagnostic_no_guard_survivor"


def test_deterministic_tie_break_prefers_simpler_candidate() -> None:
    complex_row = _candidate(
        cost=104.0, low_cost=18.0, cost_weight=0.4,
        continuation=0.25, iteration=7, tie_margin=0.03,
    )
    simple_row = _candidate(
        cost=104.0, low_cost=18.0, cost_weight=0.2,
        continuation=0.0, iteration=2, tie_margin=0.0,
    )
    selected = select_cost_aware_candidate(
        [complex_row, simple_row], _comparator(), SelectionGuards()
    )
    assert selected["cost_weight"] == 0.2
    assert selected["continuation_weight"] == 0.0
    assert selected["iteration"] == 2
    assert selected["tie_margin"] == 0.0

