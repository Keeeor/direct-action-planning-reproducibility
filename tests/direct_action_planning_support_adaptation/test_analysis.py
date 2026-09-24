from __future__ import annotations

import numpy as np
import pandas as pd

from stage2_dynamic_budget.direct_action_planning_support_adaptation.analysis import (
    METRICS,
    bh_adjust,
    paired_method_comparisons,
    pareto_cells,
    recovery_curve,
    trajectory_divergence,
)


def _metric_row(method: str, scenario: str, seed: int, regret: float) -> dict[str, object]:
    row: dict[str, object] = {
        "method": method,
        "scenario_id": scenario,
        "model_seed": seed,
        "budget": 4,
        "episode": 0,
    }
    for metric in METRICS:
        row[metric] = 0.0
    row["mean_Q_star_regret"] = regret
    row["completion_rate"] = 0.8
    row["slo_violation_rate"] = 0.2
    row["total_cost"] = 4.0
    return row


def test_bh_adjust_is_monotone_in_sorted_p_values():
    adjusted = bh_adjust(np.asarray([0.04, 0.01, 0.20]))
    assert np.allclose(adjusted, [0.06, 0.03, 0.20])


def test_paired_comparisons_use_scenario_seed_units():
    rows = []
    for scenario in ("a", "b"):
        for seed in (1, 2):
            rows.append(_metric_row("base", scenario, seed, 1.0))
            rows.append(_metric_row("candidate", scenario, seed, 0.5))
    result = paired_method_comparisons(
        pd.DataFrame(rows), baseline="base", candidates=["candidate"], family="toy"
    )
    regret = result[result.metric == "mean_Q_star_regret"].iloc[0]
    assert regret.paired_units == 4
    assert regret.mean_difference == -0.5
    assert regret.wins == 4
    assert regret.ties == 0
    assert regret.losses == 0


def test_recovery_curve_separates_zero_oracle_gain_cells():
    rows = [
        _metric_row("no_calibration", "a", 1, 1.0),
        _metric_row("oracle_target_value", "a", 1, 0.0),
        _metric_row("adapt_full_05pct", "a", 1, 0.5),
        _metric_row("no_calibration", "b", 2, 0.0),
        _metric_row("oracle_target_value", "b", 2, 0.0),
        _metric_row("adapt_full_05pct", "b", 2, 0.0),
    ]
    result = recovery_curve(pd.DataFrame(rows))
    row = result[result.method == "adapt_full_05pct"].iloc[0]
    assert row.cells == 2
    assert row.informative_cells == 1
    assert row.registered_mean_recovery == 0.75
    assert row.informative_mean_recovery == 0.5


def test_pareto_counts_budget_cells_descriptively():
    rows = [
        _metric_row("base", "a", 1, 0.0),
        _metric_row("candidate", "a", 1, 0.0),
    ]
    rows[1]["completion_rate"] = 0.9
    rows[1]["slo_violation_rate"] = 0.1
    rows[1]["total_cost"] = 3.0
    result = pareto_cells(pd.DataFrame(rows), baseline="base", candidates=["candidate"])
    assert result.iloc[0].descriptive_budget_cells == 1
    assert result.iloc[0].candidate_dominates == 1


def test_trajectory_divergence_reports_cutoff_state_shift():
    rows = []
    for method in ("oracle_target_value", "adapt_full_10pct"):
        for t in range(6):
            rows.append(
                {
                    "method": method,
                    "scenario_id": "a",
                    "model_seed": 1,
                    "test_seed": 7,
                    "budget": 4,
                    "episode": 0,
                    "eval_seed": 9,
                    "t": t,
                    "load": 0,
                    "queue": int(method.startswith("adapt") and t >= 2),
                    "remaining_budget": 4 - int(method.startswith("adapt") and t >= 1),
                    "action": int(method.startswith("adapt") and t == 1),
                }
            )
    result = trajectory_divergence(pd.DataFrame(rows), cutoff=4)
    summary = result["summary"]
    assert summary["trajectories"] == 1
    assert summary["first_divergence_t_mean"] == 1.0
    assert summary["cutoff_state_divergence_rate"] == 1.0
    assert summary["cutoff_queue_mae"] == 1.0
    assert summary["cutoff_budget_mae"] == 1.0
