from __future__ import annotations

import numpy as np
import pandas as pd

from stage2_dynamic_budget.direct_action_planning_context_value.analysis import (
    bh_adjust,
    paired_comparisons,
    pareto_cells,
)


def _episodes() -> pd.DataFrame:
    rows = []
    for scenario in ("early_burst", "late_burst", "periodic"):
        for model_seed in range(5):
            for budget in (4, 8, 12):
                for method, regret, cost in (
                    ("frozen_D1", 0.02, 8.0),
                    ("final_context_value", 0.01, 8.0),
                ):
                    rows.append(
                        {
                            "scenario": scenario,
                            "model_seed": model_seed,
                            "budget": budget,
                            "episode": 0,
                            "method": method,
                            "mean_Q_star_regret": regret,
                            "action_consistency_rate": 1.0 - regret,
                            "return_gap_to_paired_optimal": regret,
                            "budget_trajectory_mae": regret,
                            "completion_rate": 0.9,
                            "slo_violation_rate": 0.1,
                            "total_cost": cost,
                        }
                    )
    return pd.DataFrame(rows)


def test_bh_adjust_is_bounded_and_monotone_after_sorting():
    adjusted = bh_adjust(np.array([0.04, 0.001, 0.02]))
    assert np.all((0 <= adjusted) & (adjusted <= 1))
    order = np.argsort([0.04, 0.001, 0.02])
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_paired_analysis_uses_15_scenario_seed_units():
    result = paired_comparisons(_episodes(), bootstrap_draws=100, seed=3)
    row = result[
        (result.candidate == "final_context_value")
        & (result.metric == "mean_Q_star_regret")
    ].iloc[0]
    assert row.paired_units == 15
    assert row.mean_difference < 0


def test_pareto_counts_budget_cells_without_calling_them_independent_replicates():
    result = pareto_cells(_episodes())
    assert result.iloc[0].descriptive_budget_cells == 45
    assert result.iloc[0].equal == 45
