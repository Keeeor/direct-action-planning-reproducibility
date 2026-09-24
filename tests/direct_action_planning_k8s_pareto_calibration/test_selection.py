from __future__ import annotations

from stage2_dynamic_budget.direct_action_planning_k8s_pareto_calibration.selection import (
    rank_screen_candidates,
)


def test_screen_ranking_uses_all_three_noninferiority_margins() -> None:
    rows = [
        {"method": "a", "continuation_weight": 0.1, "completion_loss": 0.005, "slo_increase": 0.01, "ready_cost_increase_seconds": 8.0, "budget_violation_seconds": 0.0},
        {"method": "b", "continuation_weight": 0.2, "completion_loss": 0.02, "slo_increase": 0.0, "ready_cost_increase_seconds": -20.0, "budget_violation_seconds": 0.0},
        {"method": "c", "continuation_weight": 0.35, "completion_loss": 0.0, "slo_increase": 0.0, "ready_cost_increase_seconds": 12.0, "budget_violation_seconds": 0.0},
    ]
    ranked = rank_screen_candidates(rows, completion_loss_max=0.01, slo_increase_max=0.02, ready_cost_increase_seconds_max=10.0)
    assert [row["method"] for row in ranked] == ["a", "c", "b"]
    assert ranked[0]["passed_all_point_guards"] is True
    assert ranked[1]["passed_all_point_guards"] is False


def test_budget_violation_cannot_rank_as_a_survivor() -> None:
    rows = [{"method": "bad", "continuation_weight": 0.05, "completion_loss": 0.0, "slo_increase": -0.1, "ready_cost_increase_seconds": -100.0, "budget_violation_seconds": 1e-3}]
    ranked = rank_screen_candidates(rows, completion_loss_max=0.01, slo_increase_max=0.02, ready_cost_increase_seconds_max=10.0)
    assert ranked[0]["passed_all_point_guards"] is False
    assert ranked[0]["normalized_guard_violation"] > 0.0

