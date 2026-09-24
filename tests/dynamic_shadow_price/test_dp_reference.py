import numpy as np

from stage2_dynamic_budget.dynamic_shadow_price.dp_reference import (
    DiscreteDPConfig,
    DiscreteBudgetMDP,
    simulate_optimal_policy,
    solve_backward_dp,
)


def test_dp_policy_is_budget_feasible_and_satisfies_bellman_equation():
    config = DiscreteDPConfig(horizon=10, max_budget=9, scenario="early_burst")
    mdp = DiscreteBudgetMDP(config)
    result = solve_backward_dp(mdp)
    assert result.values.shape == (11, 3, config.max_queue + 1, 10)
    assert result.actions.shape == (10, 3, config.max_queue + 1, 10)
    assert result.max_bellman_residual < 1e-10
    for t in range(config.horizon):
        for budget in range(config.max_budget + 1):
            chosen = result.actions[t, :, :, budget]
            assert np.all(mdp.action_costs[chosen] <= budget)


def test_dp_value_is_monotone_in_budget_and_shadow_price_is_nonnegative():
    mdp = DiscreteBudgetMDP(
        DiscreteDPConfig(horizon=12, max_budget=12, scenario="late_burst")
    )
    result = solve_backward_dp(mdp)
    assert np.min(np.diff(result.values, axis=-1)) >= -1e-10
    assert np.nanmin(result.shadow_prices[..., 1:]) >= -1e-10


def test_optimal_action_can_depend_on_remaining_budget_and_horizon():
    mdp = DiscreteBudgetMDP(
        DiscreteDPConfig(horizon=12, max_budget=12, scenario="late_burst")
    )
    result = solve_backward_dp(mdp)
    budget_dependence = np.any(
        result.actions[..., 1:] != result.actions[..., :-1]
    )
    horizon_dependence = np.any(result.actions[0] != result.actions[-1])
    assert budget_dependence
    assert horizon_dependence


def test_optimal_trajectory_is_reproducible_and_respects_total_budget():
    mdp = DiscreteBudgetMDP(
        DiscreteDPConfig(horizon=12, max_budget=12, scenario="early_burst")
    )
    result = solve_backward_dp(mdp)
    left = simulate_optimal_policy(mdp, result, initial_budget=8, seed=13)
    right = simulate_optimal_policy(mdp, result, initial_budget=8, seed=13)
    assert left == right
    assert len(left) == mdp.config.horizon
    assert sum(row["cost"] for row in left) <= 8
    assert left[-1]["remaining_budget"] >= 0
