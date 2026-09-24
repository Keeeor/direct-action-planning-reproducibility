from __future__ import annotations

import numpy as np
import pandas as pd

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from stage2_dynamic_budget.direct_action_planning.learning import (
    collect_transition_samples,
    fit_empirical_action_model,
    solve_empirical_value,
)
from stage2_dynamic_budget.direct_action_planning.planning import one_step_plan
from stage2_dynamic_budget.direct_action_planning_repair.legacy_diagnostics import (
    closed_loop_first_errors,
    diagnose_state_grid,
)


def test_state_diagnostic_preserves_exact_planner_values_and_all_feasible_pairs() -> None:
    mdp = ActionConditionedBudgetMDP(
        ACBADPConfig(horizon=3, max_budget=3, max_queue=2, scenario="late_burst")
    )
    optimum = solve_action_dp(mdp)
    samples = collect_transition_samples(mdp, samples_per_state_action=64, seed=17)
    model = fit_empirical_action_model(mdp, samples, smoothing=0.25)
    value = solve_empirical_value(mdp, model)
    actions, states, pairs = diagnose_state_grid(
        mdp, optimum, value, model, "late_burst", seed=17
    )

    assert len(states) == int(np.prod(optimum.actions.shape))
    assert set(
        [
            "q_lv",
            "q_lm",
            "q_star",
            "next_load_probability_tv",
            "next_queue_mean_abs_error",
            "reward_abs_error",
            "cost_abs_error",
        ]
    ).issubset(actions.columns)
    target = actions.query("t == 1 and load == 2 and queue == 1 and remaining_budget == 3")
    lv = one_step_plan(mdp, value, t=1, load=2, queue=1, budget=3)
    assert np.allclose(target.sort_values("action").q_lv, lv.q_values, equal_nan=True)
    expected_pairs = sum(
        len(feasible) * (len(feasible) - 1) // 2
        for feasible in (
            np.flatnonzero(np.isfinite(optimum.q_values[index]))
            for index in np.ndindex(optimum.actions.shape)
        )
    )
    assert len(pairs) == expected_pairs
    assert pairs.comparable.dtype == bool


def test_closed_loop_first_error_marks_first_action_difference_and_budget_drift() -> None:
    rows = []
    for method, actions, budgets in (
        ("learned_value_branch", [0, 1, 0], [3, 2, 2]),
        ("learned_model_branch", [0, 2, 0], [3, 1, 1]),
    ):
        for t, (action, remaining) in enumerate(zip(actions, budgets)):
            rows.append(
                {
                    "method": method,
                    "scenario": "late_burst",
                    "seed": 0,
                    "budget": 3,
                    "episode": 0,
                    "eval_seed": 123,
                    "t": t,
                    "load": 1,
                    "queue": t,
                    "budget_before": 3 if t == 0 else budgets[t - 1],
                    "remaining_budget": remaining,
                    "action": action,
                    "Q_star_regret": float(action),
                }
            )
    first, divergence = closed_loop_first_errors(pd.DataFrame(rows))
    assert len(first) == 1
    assert first.iloc[0].first_error_t == 1
    assert first.iloc[0].has_action_divergence
    assert first.iloc[0].post_error_budget_mae == 1.0
    assert set(divergence.t) == {1, 2}
