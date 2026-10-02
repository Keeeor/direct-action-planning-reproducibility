from __future__ import annotations

import numpy as np

from dap.direct_action_planning_action_effect_sensitivity.planner import (
    make_capacity_biased_planner,
)
from dap.direct_action_planning_paper_closure.planning import (
    make_scaled_planner,
)
from dap.envs.synthetic_queue_env import SyntheticQueueConfig
from dap.envs.trace_driven_env import TraceDrivenQueueEnv


class Forecaster:
    def predict(self, observation):
        return float(observation[0])


class Value:
    def predict(self, observations):
        values = np.asarray(observations, dtype=np.float64)
        return values[:, 4] - 0.1 * values[:, 2]


def _env(budget: float = 8.0):
    env = TraceDrivenQueueEnv(
        np.asarray([7.0, 9.0, 12.0, 8.0], dtype=np.float64),
        SyntheticQueueConfig(horizon=4, budget=budget, scenario="gold"),
    )
    observation, _ = env.reset(seed=3)
    return env, observation


def test_factor_one_exactly_reproduces_frozen_structured_planner():
    env, observation = _env()
    reference = make_scaled_planner(
        value=Value(), forecaster=Forecaster(), gamma=0.99,
        continuation_weight=0.5,
    )
    sensitivity = make_capacity_biased_planner(
        value=Value(), forecaster=Forecaster(), gamma=0.99,
        continuation_weight=0.5, capacity_factor=1.0,
    )
    action_reference, q_reference = reference(env, observation)
    action_sensitivity, q_sensitivity = sensitivity(env, observation)
    assert action_sensitivity == action_reference
    assert np.array_equal(q_sensitivity, q_reference)


def test_capacity_error_changes_only_copied_candidate_model():
    env, observation = _env()
    original_deltas = env.capacity_deltas.copy()
    original_costs = env.action_costs.copy()
    planner = make_capacity_biased_planner(
        value=Value(), forecaster=Forecaster(), gamma=0.99,
        continuation_weight=0.5, capacity_factor=0.9,
    )
    _action, _scores = planner(env, observation)
    assert np.array_equal(env.capacity_deltas, original_deltas)
    assert np.array_equal(env.action_costs, original_costs)
    assert env.t == 0
    assert env.cumulative_cost == 0.0


def test_hard_affordability_is_identical_under_all_capacity_factors():
    for factor in (0.9, 0.95, 1.0, 1.05, 1.1):
        env, observation = _env(budget=1.0)
        planner = make_capacity_biased_planner(
            value=Value(), forecaster=Forecaster(), gamma=0.99,
            continuation_weight=0.5, capacity_factor=factor,
        )
        action, scores = planner(env, observation)
        assert action in (0, 1)
        assert np.isneginf(scores[2:]).all()


def test_invalid_capacity_factors_fail_closed():
    for factor in (0.0, -1.0, np.nan, np.inf):
        try:
            make_capacity_biased_planner(
                value=Value(), forecaster=Forecaster(), gamma=0.99,
                continuation_weight=0.5, capacity_factor=factor,
            )
        except ValueError:
            continue
        raise AssertionError(f"invalid capacity factor accepted: {factor}")
