from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch


Planner = Callable[[object, np.ndarray], tuple[int, np.ndarray]]


def feasible_actions(env) -> np.ndarray:
    remaining = max(env.config.budget - env.cumulative_cost, 0.0)
    return np.flatnonzero(env.action_costs <= remaining + 1.0e-8)


def replace_next_load_numpy(
    next_observations: np.ndarray,
    predicted_load: np.ndarray,
) -> np.ndarray:
    """Inject load and the two observation fields analytically derived from it."""

    updated = np.asarray(next_observations, dtype=np.float32).copy()
    predicted = np.asarray(predicted_load, dtype=np.float32).reshape(-1)
    if updated.ndim != 3 or updated.shape[0] != len(predicted) or updated.shape[2] != 14:
        raise ValueError("next_observations must be [states, actions, 14]")
    updated[:, :, 0] = predicted[:, None]
    updated[:, :, 10] = predicted[:, None] / np.maximum(updated[:, :, 1], 1.0e-6)
    updated[:, :, 11] = predicted[:, None] / np.maximum(updated[:, :, 4], 1.0e-6)
    return updated


def replace_next_load_tensor(
    next_observations: torch.Tensor,
    predicted_load: torch.Tensor,
) -> torch.Tensor:
    predicted = predicted_load[:, None].expand(-1, next_observations.shape[1])
    columns = [next_observations[:, :, index] for index in range(14)]
    columns[0] = predicted
    columns[10] = predicted / next_observations[:, :, 1].clamp_min(1.0e-6)
    columns[11] = predicted / next_observations[:, :, 4].clamp_min(1.0e-6)
    return torch.stack(columns, dim=-1)


def make_evidence_planner(
    *,
    value,
    forecaster,
    gamma: float,
    mode: str = "structured",
    full_transition=None,
    use_continuation: bool = True,
    forecast_multiplier: float = 1.0,
) -> Planner:
    allowed = {"structured", "persistence", "oracle_next_load", "full_transition"}
    if mode not in allowed:
        raise ValueError(f"unknown planner mode: {mode}")
    if mode == "full_transition" and full_transition is None:
        raise ValueError("full_transition mode requires a model")

    def plan(env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        q_values = np.full(len(env.action_costs), -np.inf, dtype=np.float64)
        if mode == "persistence":
            predicted_load = float(observation[0])
        elif mode == "oracle_next_load":
            next_index = env.t + 1
            predicted_load = (
                float(env._arrival_rates[next_index])
                if next_index < env.config.horizon
                else 0.0
            )
        elif mode == "structured":
            predicted_load = max(float(forecaster.predict(observation)), 0.0)
        else:
            predicted_load = 0.0
        predicted_load *= float(forecast_multiplier)

        for action in feasible_actions(env):
            if mode == "full_transition":
                with torch.no_grad():
                    state = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
                    action_tensor = torch.as_tensor([action], dtype=torch.long)
                    next_tensor, reward_tensor = full_transition(state, action_tensor)
                next_observation = np.nan_to_num(
                    next_tensor.numpy()[0], nan=0.0, posinf=1.0e3, neginf=0.0
                )
                next_observation[0] = max(next_observation[0], 0.0)
                next_observation[2] = max(next_observation[2], 0.0)
                next_observation[-2] = max(
                    observation[-2]
                    - env.action_costs[action] / max(env.config.budget, 1.0e-8),
                    0.0,
                )
                next_observation[-1] = max(
                    observation[-1] - 1.0 / env.config.horizon, 0.0
                )
                reward = float(reward_tensor.item())
                done = env.t + 1 >= env.config.horizon
            else:
                branch = copy.deepcopy(env)
                next_index = branch.t + 1
                if next_index < branch.config.horizon:
                    branch._arrival_rates[next_index] = predicted_load
                next_observation, reward, terminated, truncated, _ = branch.step(int(action))
                done = bool(terminated or truncated)
            continuation = 0.0
            if use_continuation and not done:
                continuation = float(value.predict(next_observation.reshape(1, -1))[0])
            q_values[action] = float(reward) + float(gamma) * continuation
        return int(np.argmax(q_values)), q_values

    return plan
