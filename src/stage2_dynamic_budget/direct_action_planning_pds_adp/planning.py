from __future__ import annotations

import copy
from typing import Callable

import numpy as np


Planner = Callable[[object, np.ndarray], tuple[int, np.ndarray]]
DEFAULT_EXOGENOUS_INDICES = (0, 10, 11)


def postdecision_state(
    next_observation: np.ndarray,
    *,
    exogenous_indices: tuple[int, ...] = DEFAULT_EXOGENOUS_INDICES,
) -> np.ndarray:
    """Remove next-load information from an action-controlled afterstate.

    The simulator has already applied the current action, reward, queue update,
    and budget decrement.  Its returned observation also exposes the next
    exogenous load.  A post-decision state must precede that revelation, so the
    three fields derived from next load are neutralized while all endogenous
    queue/resource/budget/history fields are retained.
    """

    values = np.asarray(next_observation)
    if values.ndim < 1 or values.shape[-1] != 14:
        raise ValueError("next_observation must have 14 features on the last axis")
    indices = tuple(int(index) for index in exogenous_indices)
    if len(set(indices)) != len(indices) or any(index < 0 or index >= 14 for index in indices):
        raise ValueError("exogenous_indices must be unique indices in [0, 14)")
    transformed = values.copy()
    transformed[..., list(indices)] = 0.0
    return transformed


def _feasible_actions(env) -> np.ndarray:
    remaining = max(float(env.config.budget) - float(env.cumulative_cost), 0.0)
    return np.flatnonzero(np.asarray(env.action_costs) <= remaining + 1.0e-8)


def make_postdecision_planner(
    *,
    value,
    gamma: float,
    continuation_weight: float,
    exogenous_indices: tuple[int, ...] = DEFAULT_EXOGENOUS_INDICES,
) -> Planner:
    """Select a feasible action using immediate reward plus PDS value."""

    discount = float(gamma)
    weight = float(continuation_weight)
    if not np.isfinite(discount) or discount < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("continuation_weight must be finite and in [0, 1]")

    def plan(env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        del observation
        q_values = np.full(len(env.action_costs), -np.inf, dtype=np.float64)
        for action in _feasible_actions(env):
            branch = copy.deepcopy(env)
            next_observation, reward, terminated, truncated, _ = branch.step(int(action))
            continuation = 0.0
            if weight > 0.0 and not (terminated or truncated):
                afterstate = postdecision_state(
                    next_observation, exogenous_indices=exogenous_indices
                )
                continuation = float(value.predict(afterstate.reshape(1, -1))[0])
            q_values[action] = float(reward) + discount * weight * continuation
        return int(np.argmax(q_values)), q_values

    return plan
