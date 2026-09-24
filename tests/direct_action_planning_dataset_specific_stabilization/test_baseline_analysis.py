from __future__ import annotations

import numpy as np
import pandas as pd

from stage2_dynamic_budget.direct_action_planning_dataset_specific_stabilization.baseline_analysis import (
    bh_fdr,
    budget_sensitivity,
    compute_unit_metrics,
    pareto_counts,
    seed_block_difference,
)


def test_unit_metrics_keep_seed_as_independent_block() -> None:
    rows = []
    for seed in (1, 2):
        for episode, value in enumerate((1.0, 3.0)):
            rows.append(
                {
                    "dataset": "d",
                    "budget": 8.0,
                    "training_seed": seed,
                    "method": "m",
                    "episode": episode,
                    "discounted_return": value + seed,
                    "completion_ratio": value / 4.0,
                    "slo_violation_rate": 1.0 - value / 4.0,
                    "total_cost": value,
                    "queue_area": value * 2.0,
                    "decision_ms_mean": 0.1,
                    "decision_ms_p95": 0.2,
                    "budget_overspend": 0.0,
                }
            )
    units = compute_unit_metrics(pd.DataFrame(rows))
    assert len(units) == 2
    assert set(units.training_seed) == {1, 2}
    assert units.return_cvar20.tolist() == [2.0, 3.0]


def test_seed_block_difference_averages_repeated_budgets() -> None:
    rows = []
    for seed, differences in ((1, (1.0, 3.0)), (2, (2.0, 4.0))):
        for budget, difference in zip((8.0, 16.0), differences, strict=True):
            rows.append(
                {
                    "dataset": "d",
                    "budget": budget,
                    "training_seed": seed,
                    "method": "dap",
                    "discounted_return": 10.0 + difference,
                }
            )
            rows.append(
                {
                    "dataset": "d",
                    "budget": budget,
                    "training_seed": seed,
                    "method": "base",
                    "discounted_return": 10.0,
                }
            )
    values = seed_block_difference(
        pd.DataFrame(rows),
        dataset="d",
        baseline="base",
        metric="discounted_return",
    )
    np.testing.assert_allclose(values, [2.0, 3.0])


def test_bh_fdr_is_monotone_in_sorted_p_values() -> None:
    corrected = bh_fdr(np.asarray([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(corrected, [0.03, 0.04, 0.04])


def test_pareto_counts_distinguish_dominance_and_tradeoff() -> None:
    result = pareto_counts(
        np.asarray([2.0, 2.0, 1.0]),
        np.asarray([1.0, 3.0, 2.0]),
        np.asarray([1.0, 2.0, 1.0]),
        np.asarray([2.0, 1.0, 2.0]),
    )
    assert result == {
        "dap_dominates": 1,
        "baseline_dominates": 1,
        "ties": 0,
        "tradeoffs": 1,
    }


def test_budget_sensitivity_reports_each_budget_without_pooling() -> None:
    rows = []
    for budget, delta in ((8.0, 1.0), (16.0, -1.0)):
        for seed in (1, 2):
            for method, value in (("dap_calibrated", 10.0 + delta), ("base", 10.0)):
                rows.append(
                    {
                        "dataset": "d",
                        "budget": budget,
                        "training_seed": seed,
                        "method": method,
                        "discounted_return": value,
                        "completion_ratio": value,
                        "slo_violation_rate": 0.0,
                        "total_cost": 1.0,
                    }
                )
    result = budget_sensitivity(pd.DataFrame(rows))
    assert result[result.budget.eq(8.0)].return_wins.item() == 2
    assert result[result.budget.eq(16.0)].return_wins.item() == 0
