"""Planning-only capacity-effect perturbations for DAP sensitivity analysis."""

from __future__ import annotations

import copy

import numpy as np


def _feasible_actions(env) -> np.ndarray:
    remaining = max(float(env.config.budget) - float(env.cumulative_cost), 0.0)
    return np.flatnonzero(env.action_costs <= remaining + 1.0e-8)


def make_capacity_biased_planner(
    *,
    value,
    forecaster,
    gamma: float,
    continuation_weight: float,
    capacity_factor: float,
):
    """Build DAP with a biased internal action-capacity model.

    The live/evaluation environment is never modified.  Only the copied
    candidate branch receives the multiplicative capacity perturbation; action
    costs, hard feasibility, the common load forecast, the value checkpoint,
    and the delivered-action interface are unchanged.
    """

    factor = float(capacity_factor)
    weight = float(continuation_weight)
    discount = float(gamma)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("capacity_factor must be finite and positive")
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("continuation_weight must be finite and in [0, 1]")
    if not np.isfinite(discount) or discount < 0.0:
        raise ValueError("gamma must be finite and non-negative")

    def plan(env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        q_values = np.full(len(env.action_costs), -np.inf, dtype=np.float64)
        predicted_load = max(float(forecaster.predict(observation)), 0.0)
        for action in _feasible_actions(env):
            branch = copy.deepcopy(env)
            # The no-op remains no-op.  Every nonzero known capacity effect is
            # perturbed by the same registered factor inside the planner only.
            branch.capacity_deltas = (
                np.asarray(branch.capacity_deltas, dtype=np.float64) * factor
            )
            next_index = branch.t + 1
            if next_index < branch.config.horizon:
                branch._arrival_rates[next_index] = predicted_load
            next_observation, reward, terminated, truncated, _ = branch.step(
                int(action)
            )
            continuation = 0.0
            if weight > 0.0 and not (terminated or truncated):
                continuation = float(
                    value.predict(next_observation.reshape(1, -1))[0]
                )
            q_values[action] = float(reward) + discount * weight * continuation
        return int(np.argmax(q_values)), q_values

    return plan
