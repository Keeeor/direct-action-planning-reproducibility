import pytest

from dap.envs.synthetic_queue_env import (
    DynamicBudgetSchedulingEnv,
    SyntheticQueueConfig,
)


def test_cumulative_and_remaining_budget_are_exact() -> None:
    env = DynamicBudgetSchedulingEnv(SyntheticQueueConfig(horizon=4, budget=10.0))
    env.reset(seed=7, options={"arrival_trace": [2.0, 2.0, 2.0, 2.0]})
    costs = []
    for action in [1, 2, 0, 3]:
        _, _, _, _, info = env.step(action)
        costs.append(info["resource_cost"])
        assert info["cumulative_cost"] == pytest.approx(sum(costs))
        assert info["remaining_budget"] == pytest.approx(max(10.0 - sum(costs), 0.0))
        assert info["remaining_budget_ratio"] == pytest.approx(
            max(10.0 - sum(costs), 0.0) / 10.0
        )


def test_zero_budget_ratio_is_finite() -> None:
    env = DynamicBudgetSchedulingEnv(SyntheticQueueConfig(horizon=1, budget=0.0))
    obs, _ = env.reset(seed=1, options={"arrival_trace": [0.0]})
    assert obs[-2] == pytest.approx(0.0)
