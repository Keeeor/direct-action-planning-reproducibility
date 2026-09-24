from __future__ import annotations

import copy

import numpy as np

from stage2_dynamic_budget.direct_action_planning_paper_closure.planning import (
    make_scaled_planner,
)
from stage2_dynamic_budget.envs.synthetic_queue_env import SyntheticQueueConfig
from stage2_dynamic_budget.envs.trace_driven_env import TraceDrivenQueueEnv


class ConstantForecaster:
    def predict(self, observation: np.ndarray) -> float:
        del observation
        return 7.0


class TerminalValueMustNotBeCalled:
    def predict(self, observations: np.ndarray) -> np.ndarray:
        del observations
        raise AssertionError("terminal continuation value was evaluated")


def test_last_trace_decision_uses_immediate_reward_only() -> None:
    env = TraceDrivenQueueEnv(
        np.asarray([7.0]),
        SyntheticQueueConfig(horizon=1, budget=4.0, scenario="terminal-mask"),
    )
    observation, _ = env.reset(seed=7)
    expected = np.full(len(env.action_costs), -np.inf, dtype=np.float64)
    for action in np.flatnonzero(env.action_costs <= env.config.budget):
        branch = copy.deepcopy(env)
        _, reward, terminated, truncated, _ = branch.step(int(action))
        assert terminated and not truncated
        expected[action] = reward

    planner = make_scaled_planner(
        value=TerminalValueMustNotBeCalled(),
        forecaster=ConstantForecaster(),
        gamma=0.99,
        continuation_weight=1.0,
    )
    action, scores = planner(env, observation)

    np.testing.assert_allclose(scores, expected, rtol=0.0, atol=1.0e-12)
    assert action == int(np.argmax(expected))
