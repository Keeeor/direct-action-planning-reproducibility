import math

import pytest

from stage2_dynamic_budget.evaluation.metrics import summarize_episode


def test_episode_metrics_match_small_fixture() -> None:
    rows = [
        {"reward": 1.0, "resource_cost": 0.0, "queue_length": 4.0, "mean_latency": 2.0, "tail_latency": 3.0, "slo_violation": 0, "arrivals": 5.0, "served": 4.0, "risk_level": 0.2, "local_budget": 0.5},
        {"reward": -1.0, "resource_cost": 2.0, "queue_length": 8.0, "mean_latency": 4.0, "tail_latency": 7.0, "slo_violation": 1, "arrivals": 5.0, "served": 5.0, "risk_level": 0.8, "local_budget": 2.0},
    ]
    summary = summarize_episode(rows, budget=4.0, horizon=2)
    assert summary["episode_reward"] == pytest.approx(0.0)
    assert summary["total_cost"] == pytest.approx(2.0)
    assert summary["slo_violation_rate"] == pytest.approx(0.5)
    assert summary["completion_rate"] == pytest.approx(0.9)
    assert summary["mean_queue"] == pytest.approx(6.0)
    assert summary["budget_utilization"] == pytest.approx(0.5)
    assert summary["budget_reallocation_ratio"] == pytest.approx(4.0)
    assert math.isfinite(summary["risk_budget_correlation"])


def test_missing_local_budget_reports_nan_mechanism_metrics() -> None:
    rows = [{"reward": 0.0, "resource_cost": 0.0, "queue_length": 0.0, "mean_latency": 1.0, "tail_latency": 1.0, "slo_violation": 0, "arrivals": 0.0, "served": 0.0, "risk_level": 0.0}]
    summary = summarize_episode(rows, budget=1.0, horizon=1)
    assert math.isnan(summary["risk_budget_correlation"])
    assert math.isnan(summary["budget_reallocation_ratio"])
