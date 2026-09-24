from __future__ import annotations

import numpy as np

from stage2_dynamic_budget.direct_action_planning_pds_adp.planning import (
    make_postdecision_planner,
    postdecision_state,
)
from stage2_dynamic_budget.envs.synthetic_queue_env import (
    DynamicBudgetSchedulingEnv,
    SyntheticQueueConfig,
)


class CapacityValue:
    def predict(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float64)
        return 100.0 * values[:, 4]


def test_postdecision_state_removes_only_next_load_derived_fields():
    observations = np.arange(28, dtype=np.float32).reshape(2, 14)
    transformed = postdecision_state(observations, exogenous_indices=(0, 10, 11))
    assert np.all(transformed[:, [0, 10, 11]] == 0.0)
    np.testing.assert_array_equal(transformed[:, 1:10], observations[:, 1:10])
    np.testing.assert_array_equal(transformed[:, 12:], observations[:, 12:])
    np.testing.assert_array_equal(observations, np.arange(28, dtype=np.float32).reshape(2, 14))


def test_postdecision_planner_honors_hard_budget_mask():
    env = DynamicBudgetSchedulingEnv(
        SyntheticQueueConfig(horizon=2, budget=1.0, scenario="stable")
    )
    observation, _ = env.reset(seed=7, options={"arrival_trace": np.asarray([6.0, 6.0])})
    planner = make_postdecision_planner(
        value=CapacityValue(), gamma=0.99, continuation_weight=1.0
    )
    action, q_values = planner(env, observation)
    assert action == 1
    assert np.isfinite(q_values[:2]).all()
    assert np.isneginf(q_values[2:]).all()


def test_postdecision_planner_does_not_read_future_load():
    config = SyntheticQueueConfig(horizon=2, budget=4.0, scenario="stable")
    planners = []
    for future in (1.0, 100.0):
        env = DynamicBudgetSchedulingEnv(config)
        observation, _ = env.reset(
            seed=3, options={"arrival_trace": np.asarray([6.0, future])}
        )
        planner = make_postdecision_planner(
            value=CapacityValue(), gamma=0.99, continuation_weight=1.0
        )
        planners.append(planner(env, observation))
    assert planners[0][0] == planners[1][0]
    np.testing.assert_allclose(planners[0][1], planners[1][1])
