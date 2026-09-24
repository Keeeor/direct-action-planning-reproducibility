from __future__ import annotations

import copy
from typing import Callable

import numpy as np


Planner = Callable[[object, np.ndarray], tuple[int, np.ndarray]]


def _feasible_actions(env) -> np.ndarray:
    remaining = max(env.config.budget - env.cumulative_cost, 0.0)
    return np.flatnonzero(env.action_costs <= remaining + 1.0e-8)


def make_scaled_planner(
    *,
    value,
    forecaster,
    gamma: float,
    continuation_weight: float,
) -> Planner:
    """Build a structured one-step planner with a frozen continuation weight."""

    weight = float(continuation_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("continuation_weight must be finite and in [0, 1]")
    discount = float(gamma)
    if not np.isfinite(discount) or discount < 0.0:
        raise ValueError("gamma must be finite and non-negative")

    def plan(env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        q_values = np.full(len(env.action_costs), -np.inf, dtype=np.float64)
        predicted_load = max(float(forecaster.predict(observation)), 0.0)
        for action in _feasible_actions(env):
            branch = copy.deepcopy(env)
            next_index = branch.t + 1
            if next_index < branch.config.horizon:
                branch._arrival_rates[next_index] = predicted_load
            next_observation, reward, terminated, truncated, _ = branch.step(int(action))
            continuation = 0.0
            if weight > 0.0 and not (terminated or truncated):
                continuation = float(value.predict(next_observation.reshape(1, -1))[0])
            q_values[action] = float(reward) + discount * weight * continuation
        return int(np.argmax(q_values)), q_values

    return plan

