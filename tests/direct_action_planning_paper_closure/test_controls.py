from __future__ import annotations

import numpy as np

from stage2_dynamic_budget.direct_action_planning_paper_closure.controls import (
    make_mpc_planner,
)
from stage2_dynamic_budget.envs.synthetic_queue_env import (
    SyntheticQueueConfig,
)
from stage2_dynamic_budget.envs.trace_driven_env import TraceDrivenQueueEnv


class ConstantForecaster:
    def predict(self, observation):
        del observation
        return 7.0


def test_mpc_masks_unaffordable_actions_and_returns_all_scores():
    env = TraceDrivenQueueEnv(
        np.full(8, 7.0),
        SyntheticQueueConfig(horizon=8, budget=1.0, scenario="test"),
    )
    observation, _ = env.reset(seed=3)
    planner = make_mpc_planner(
        forecaster=ConstantForecaster(), gamma=0.99, horizon=4, beam_width=8
    )
    action, scores = planner(env, observation)
    assert action in (0, 1)
    assert scores.shape == (4,)
    assert np.isneginf(scores[2:]).all()


def test_mpc_is_deterministic_for_same_state():
    config = SyntheticQueueConfig(horizon=8, budget=8.0, scenario="test")
    env_a = TraceDrivenQueueEnv(np.full(8, 7.0), config)
    env_b = TraceDrivenQueueEnv(np.full(8, 7.0), config)
    observation_a, _ = env_a.reset(seed=7)
    observation_b, _ = env_b.reset(seed=7)
    planner = make_mpc_planner(
        forecaster=ConstantForecaster(), gamma=0.99, horizon=4
    )
    action_a, scores_a = planner(env_a, observation_a)
    action_b, scores_b = planner(env_b, observation_b)
    assert action_a == action_b
    assert np.array_equal(scores_a, scores_b)
