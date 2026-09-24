from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch

from .models import FullTransitionNetwork, LoadForecaster, ValueNetwork


Planner = Callable[[object, np.ndarray], tuple[int, np.ndarray]]


def _feasible(env) -> np.ndarray:
    remaining = max(env.config.budget - env.cumulative_cost, 0.0)
    return np.flatnonzero(env.action_costs <= remaining + 1.0e-8)


def make_planner(
    method: str,
    value: ValueNetwork,
    forecaster: LoadForecaster,
    full_transition: FullTransitionNetwork,
    *,
    gamma: float,
) -> Planner:
    allowed = {
        "learned_value_true_transition",
        "original_learned_model",
        "structured_dap",
        "structured_dap_refresh",
    }
    if method not in allowed:
        raise ValueError(f"unknown planner method: {method}")

    def plan(env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        q_values = np.full(4, -np.inf, dtype=np.float64)
        predicted_load = forecaster.predict(observation)
        for action in _feasible(env):
            if method == "original_learned_model":
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
                    action_tensor = torch.as_tensor([action], dtype=torch.long)
                    next_tensor, reward_tensor = full_transition(obs_tensor, action_tensor)
                next_observation = next_tensor.numpy()[0]
                next_observation[:12] = np.nan_to_num(
                    next_observation[:12], nan=0.0, posinf=1.0e3, neginf=0.0
                )
                next_observation[0] = max(next_observation[0], 0.0)
                next_observation[2] = max(next_observation[2], 0.0)
                next_observation[-2] = max(
                    observation[-2] - env.action_costs[action] / max(env.config.budget, 1.0e-8),
                    0.0,
                )
                next_observation[-1] = max(
                    observation[-1] - 1.0 / env.config.horizon,
                    0.0,
                )
                reward = float(reward_tensor.item())
                done = env.t + 1 >= env.config.horizon
            else:
                branch = copy.deepcopy(env)
                if method in {"structured_dap", "structured_dap_refresh"}:
                    next_index = branch.t + 1
                    if next_index < branch.config.horizon:
                        branch._arrival_rates[next_index] = predicted_load
                next_observation, reward, terminated, truncated, _ = branch.step(int(action))
                done = bool(terminated or truncated)
            continuation = 0.0 if done else float(value.predict(next_observation.reshape(1, -1))[0])
            q_values[action] = reward + gamma * continuation
        return int(np.argmax(q_values)), q_values

    return plan


def assert_structured_planner_prefix_invariant(
    planner: Planner,
    env_a,
    env_b,
) -> None:
    observation_a, _ = env_a.reset(seed=0)
    observation_b, _ = env_b.reset(seed=0)
    np.testing.assert_allclose(observation_a, observation_b)
    action_a, q_a = planner(env_a, observation_a)
    action_b, q_b = planner(env_b, observation_b)
    if action_a != action_b:
        raise AssertionError("structured planner action depends on unseen future trace")
    np.testing.assert_allclose(q_a, q_b, rtol=1.0e-6, atol=1.0e-6)
