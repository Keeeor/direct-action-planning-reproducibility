from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def feasible_actions(env) -> np.ndarray:
    remaining = max(float(env.config.budget) - float(env.cumulative_cost), 0.0)
    return np.flatnonzero(env.action_costs <= remaining + 1.0e-8)


class Controller:
    name = "controller"

    def reset(self) -> None:
        pass

    def select(self, env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        raise NotImplementedError


class NoInterventionController(Controller):
    name = "no_intervention"

    def select(self, env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        del observation
        actions = feasible_actions(env)
        q = np.full(len(env.action_costs), -np.inf, dtype=np.float64)
        q[actions] = 0.0
        return 0 if 0 in actions else int(actions[0]), q


class MaxFeasibleController(Controller):
    name = "max_feasible"

    def select(self, env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        del observation
        actions = feasible_actions(env)
        q = np.full(len(env.action_costs), -np.inf, dtype=np.float64)
        q[actions] = env.capacity_deltas[actions]
        return int(actions[np.argmax(env.capacity_deltas[actions])]), q


@dataclass
class ReactiveThresholdController(Controller):
    """Validation-tuned threshold controller with one-step hysteresis."""

    queue_threshold: float = 8.0
    load_threshold: float = 1.15
    slo_threshold: float = 0.10
    hysteresis: float = 0.05
    name: str = "reactive_threshold"

    def reset(self) -> None:
        self._last_action = 0

    def select(self, env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        obs = np.asarray(observation, dtype=np.float64)
        score = (
            max(obs[2] - self.queue_threshold, 0.0) / max(self.queue_threshold, 1.0)
            + max(obs[0] / max(obs[4], 1.0) - self.load_threshold, 0.0)
            + max(obs[9] - self.slo_threshold, 0.0) * 2.0
        )
        if score <= self.hysteresis:
            proposed = max(self._last_action - 1, 0)
        else:
            proposed = min(int(np.floor(score * 2.0)) + 1, len(env.action_costs) - 1)
        actions = feasible_actions(env)
        action = int(actions[actions <= proposed][-1]) if np.any(actions <= proposed) else int(actions[0])
        self._last_action = action
        q = np.full(len(env.action_costs), -np.inf, dtype=np.float64)
        q[actions] = -np.abs(actions - proposed)
        return action, q


@dataclass
class PIDBudgetController(Controller):
    """Queue-pressure PID with budget pacing and anti-windup."""

    kp: float = 0.55
    ki: float = 0.04
    kd: float = 0.15
    queue_target: float = 2.0
    budget_weight: float = 1.0
    name: str = "pid_budget_autoscaler"

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = 0.0

    def select(self, env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        obs = np.asarray(observation, dtype=np.float64)
        remaining_steps = max(float(env.config.horizon - env.t), 1.0)
        remaining_budget = max(float(env.config.budget) - float(env.cumulative_cost), 0.0)
        desired_rate = remaining_budget / remaining_steps
        error = max(obs[2] - self.queue_target, 0.0) + max(obs[0] - obs[4], 0.0) * 0.25
        candidate_integral = np.clip(self._integral + error, -100.0, 100.0)
        derivative = error - self._previous_error
        control = self.kp * error + self.ki * candidate_integral + self.kd * derivative
        control += self.budget_weight * (desired_rate - float(env.action_costs[0]))
        actions = feasible_actions(env)
        target = int(np.clip(np.round(control), 0, len(env.action_costs) - 1))
        action = int(actions[actions <= target][-1]) if np.any(actions <= target) else int(actions[0])
        # Only integrate when the selected action does not saturate in the direction of error.
        if action == target or (target > action and error <= 0.0):
            self._integral = float(candidate_integral)
        self._previous_error = float(error)
        q = np.full(len(env.action_costs), -np.inf, dtype=np.float64)
        q[actions] = -np.abs(env.capacity_deltas[actions] - control)
        return action, q


def _reward_for(queue: float, arrivals: float, capacity: float, config) -> tuple[float, float]:
    available = max(queue + arrivals, 0.0)
    served = min(available, capacity)
    next_queue = min(max(available - served, 0.0), config.max_queue)
    tail_latency = 1.0 + 2.0 * next_queue / max(capacity, 1.0e-8)
    slo = float(tail_latency > config.slo_latency)
    reward = (
        config.reward_completion * served / max(available, 1.0)
        - config.reward_queue_penalty * next_queue
        - config.reward_latency_penalty * tail_latency
        - config.reward_slo_penalty * slo
    )
    return float(reward), float(next_queue)


@dataclass
class CausalMPCController(Controller):
    """Short-horizon model-predictive controller using only causal observations."""

    horizon: int = 4
    forecast_decay: float = 0.65
    cost_weight: float = 0.15
    beam_width: int = 8
    name: str = "causal_mpc"

    def select(self, env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        obs = np.asarray(observation, dtype=np.float64)
        actions = feasible_actions(env)
        q = np.full(len(env.action_costs), -np.inf, dtype=np.float64)
        current_load = max(float(obs[0]), 0.0)
        recent_load = max(float(obs[1]), 0.0)
        remaining_budget = max(float(env.config.budget) - float(env.cumulative_cost), 0.0)
        initial_queue = float(max(obs[2], 0.0))
        for first_action in actions:
            capacity = float(env.config.base_capacity + env.capacity_deltas[first_action])
            reward, next_queue = _reward_for(initial_queue, current_load, capacity, env.config)
            first_cost = float(env.action_costs[first_action])
            beam = [(reward - self.cost_weight * first_cost, next_queue, remaining_budget - first_cost, current_load)]
            for depth in range(1, self.horizon):
                expanded = []
                for score, queue, budget_left, forecast in beam:
                    next_forecast = self.forecast_decay * forecast + (1.0 - self.forecast_decay) * recent_load
                    future_actions = np.flatnonzero(env.action_costs <= budget_left + 1.0e-8)
                    for future_action in future_actions:
                        future_capacity = float(env.config.base_capacity + env.capacity_deltas[future_action])
                        future_reward, future_queue = _reward_for(queue, next_forecast, future_capacity, env.config)
                        future_cost = float(env.action_costs[future_action])
                        expanded.append((
                            score + (0.99**depth) * future_reward - self.cost_weight * future_cost,
                            future_queue,
                            budget_left - future_cost,
                            next_forecast,
                        ))
                expanded.sort(key=lambda item: item[0], reverse=True)
                beam = expanded[: self.beam_width]
            q[int(first_action)] = max(item[0] for item in beam)
        return int(np.argmax(q)), q


@dataclass
class LyapunovDPPController(Controller):
    """Queue drift-plus-penalty rule with a virtual budget deficit queue."""

    penalty_weight: float = 0.35
    queue_weight: float = 1.0
    name: str = "lyapunov_dpp"

    def reset(self) -> None:
        self._virtual_cost_queue = 0.0

    def select(self, env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        obs = np.asarray(observation, dtype=np.float64)
        actions = feasible_actions(env)
        target_rate = float(env.config.budget) / max(float(env.config.horizon), 1.0)
        q = np.full(len(env.action_costs), -np.inf, dtype=np.float64)
        for action in actions:
            capacity = float(env.config.base_capacity + env.capacity_deltas[action])
            served = min(max(float(obs[2]) + float(obs[0]), 0.0), capacity)
            next_queue = max(float(obs[2]) + float(obs[0]) - served, 0.0)
            drift = self.queue_weight * (next_queue**2 - max(float(obs[2]), 0.0) ** 2)
            cost_deficit = self._virtual_cost_queue + float(env.action_costs[action]) - target_rate
            q[action] = -drift - self.penalty_weight * cost_deficit * float(env.action_costs[action])
        action = int(np.argmax(q))
        # Update the virtual queue using the selected action's actual cost after selection.
        self._virtual_cost_queue = max(
            0.0,
            self._virtual_cost_queue + float(env.action_costs[action]) - target_rate,
        )
        return action, q
