"""Same-information planning controls for the DAP attribution experiment."""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np


Planner = Callable[[object, np.ndarray], tuple[int, np.ndarray]]


def feasible_actions(env) -> np.ndarray:
    remaining = max(float(env.config.budget - env.cumulative_cost), 0.0)
    return np.flatnonzero(env.action_costs <= remaining + 1.0e-8)


def make_mpc_planner(
    *,
    forecaster,
    gamma: float,
    horizon: int,
    beam_width: int = 16,
) -> Planner:
    """Structured finite-horizon MPC with a shared exogenous-load rollout.

    The load path is produced once from a no-op reference branch. Every action
    sequence sees that same path; only known action effects differ. The terminal
    value is exactly zero, so this is not a hidden DAP value-function variant.
    """

    depth = int(horizon)
    if depth <= 0:
        raise ValueError("MPC horizon must be positive")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    discount = float(gamma)
    if not np.isfinite(discount) or discount < 0.0:
        raise ValueError("gamma must be finite and non-negative")

    def load_path(env, observation: np.ndarray) -> list[float]:
        reference = copy.deepcopy(env)
        current = observation.copy()
        values: list[float] = []
        for _ in range(depth):
            values.append(max(float(forecaster.predict(current)), 0.0))
            if reference.t + 1 >= reference.config.horizon:
                break
            reference._arrival_rates[reference.t + 1] = values[-1]
            current, _, terminated, truncated, _ = reference.step(0)
            if terminated or truncated:
                break
        return values

    def branch_step(env, action: int, predicted_load: float):
        branch = copy.deepcopy(env)
        if branch.t + 1 < branch.config.horizon:
            branch._arrival_rates[branch.t + 1] = predicted_load
        _, reward, terminated, truncated, _ = branch.step(action)
        return branch, float(reward), bool(terminated or truncated)

    def plan(env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        path = load_path(env, observation)
        q_values = np.full(len(env.action_costs), -np.inf, dtype=np.float64)
        for first in feasible_actions(env):
            first_env, reward, done = branch_step(env, int(first), path[0])
            beams: list[tuple[object, float, bool]] = [(first_env, reward, done)]
            for offset in range(1, len(path)):
                expanded: list[tuple[object, float, bool]] = []
                for branch, score, branch_done in beams:
                    if branch_done:
                        expanded.append((branch, score, True))
                        continue
                    for action in feasible_actions(branch):
                        next_env, step_reward, next_done = branch_step(
                            branch, int(action), path[offset]
                        )
                        expanded.append(
                            (next_env, score + discount**offset * step_reward, next_done)
                        )
                expanded.sort(key=lambda row: row[1], reverse=True)
                beams = expanded[:beam_width]
            q_values[first] = max(score for _, score, _ in beams)
        return int(np.argmax(q_values)), q_values

    return plan

