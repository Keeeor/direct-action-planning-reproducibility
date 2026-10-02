from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ACBADPConfig:
    horizon: int = 16
    max_budget: int = 12
    max_queue: int = 6
    scenario: str = "early_burst"
    gamma: float = 0.99
    action_costs: tuple[int, ...] = (0, 1, 2, 3)
    action_capacity: tuple[int, ...] = (1, 2, 3, 4)
    action_activation_penalty: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    load_arrivals: tuple[int, ...] = (1, 2, 4)
    queue_penalty: float = 0.45
    severe_queue_penalty: float = 1.5

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.max_budget < 0 or self.max_queue <= 0:
            raise ValueError("invalid horizon, budget, or queue range")
        if self.scenario not in {"stable", "early_burst", "late_burst", "periodic"}:
            raise ValueError("unknown DP scenario")
        if len(self.action_costs) < 2 or len(self.action_costs) != len(self.action_capacity):
            raise ValueError("action costs and capacities must have equal non-trivial length")
        if len(self.action_activation_penalty) != len(self.action_costs):
            raise ValueError("action activation penalties must match the action dimension")
        if self.action_costs[0] != min(self.action_costs):
            raise ValueError("action zero must be a cheapest reference action")
        if any(cost < 0 for cost in self.action_costs):
            raise ValueError("action costs must be non-negative")
        if any(capacity <= 0 for capacity in self.action_capacity):
            raise ValueError("action capacities must be positive")
        if any(penalty < 0 for penalty in self.action_activation_penalty):
            raise ValueError("action activation penalties must be non-negative")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")


class ActionConditionedBudgetMDP:
    """Finite-horizon queue MDP with explicit action-conditioned budget values."""

    def __init__(self, config: ACBADPConfig):
        self.config = config
        self.action_costs = np.asarray(config.action_costs, dtype=np.int64)
        self.action_capacity = np.asarray(config.action_capacity, dtype=np.int64)
        self.load_arrivals = np.asarray(config.load_arrivals, dtype=np.int64)
        self.n_loads = len(config.load_arrivals)
        self.n_actions = len(config.action_costs)

    def load_probabilities(self, t: int, current_load: int) -> np.ndarray:
        if not 0 <= current_load < self.n_loads:
            raise ValueError("invalid load state")
        persistence = np.full(self.n_loads, 0.1, dtype=np.float64)
        persistence[current_load] = 0.8
        phase = t / max(self.config.horizon - 1, 1)
        target = np.asarray([0.05, 0.15, 0.80], dtype=np.float64)
        if self.config.scenario == "stable":
            target = np.asarray([0.15, 0.70, 0.15], dtype=np.float64)
            mix = 0.25
        elif self.config.scenario == "periodic":
            periodic_peak = 0.5 * (1.0 + np.sin(6.0 * np.pi * phase - np.pi / 2.0))
            mix = 0.10 + 0.55 * periodic_peak
        else:
            center = 0.25 if self.config.scenario == "early_burst" else 0.75
            burst = np.exp(-0.5 * ((phase - center) / 0.11) ** 2)
            mix = 0.65 * burst
        probabilities = (1.0 - mix) * persistence + mix * target
        return probabilities / probabilities.sum()

    def outcome(self, queue: int, load: int, action: int) -> tuple[int, float, dict[str, float]]:
        if not 0 <= action < self.n_actions:
            raise ValueError("invalid action")
        arrivals = int(self.load_arrivals[load])
        available = queue + arrivals
        served = min(available, int(self.action_capacity[action]))
        next_queue = min(max(available - served, 0), self.config.max_queue)
        violation = float(next_queue >= max(3, self.config.max_queue // 2))
        reward = (
            float(served)
            - self.config.queue_penalty * float(next_queue)
            - self.config.severe_queue_penalty * violation
            - float(self.config.action_activation_penalty[action])
        )
        return next_queue, reward, {
            "served": float(served),
            "queue": float(next_queue),
            "slo_violation": violation,
            "cost": float(self.action_costs[action]),
        }

    def transitions(
        self, t: int, load: int, queue: int, action: int
    ) -> Iterable[tuple[float, int, int, float]]:
        next_queue, reward, _ = self.outcome(queue, load, action)
        for next_load, probability in enumerate(self.load_probabilities(t, load)):
            if probability > 0:
                yield float(probability), next_load, next_queue, reward


@dataclass(frozen=True)
class ActionDPResult:
    values: np.ndarray
    actions: np.ndarray
    q_values: np.ndarray
    advantages: np.ndarray
    max_bellman_residual: float


def _action_value(
    mdp: ActionConditionedBudgetMDP,
    values: np.ndarray,
    t: int,
    load: int,
    queue: int,
    budget: int,
    action: int,
) -> float:
    cost = int(mdp.action_costs[action])
    if cost > budget:
        return float("nan")
    expected = 0.0
    for probability, next_load, next_queue, reward in mdp.transitions(
        t, load, queue, action
    ):
        expected += probability * (
            reward
            + mdp.config.gamma
            * values[t + 1, next_load, next_queue, budget - cost]
        )
    return float(expected)


def solve_action_dp(mdp: ActionConditionedBudgetMDP) -> ActionDPResult:
    cfg = mdp.config
    values = np.zeros(
        (cfg.horizon + 1, mdp.n_loads, cfg.max_queue + 1, cfg.max_budget + 1),
        dtype=np.float64,
    )
    actions = np.zeros(
        (cfg.horizon, mdp.n_loads, cfg.max_queue + 1, cfg.max_budget + 1),
        dtype=np.int64,
    )
    q_values = np.full(actions.shape + (mdp.n_actions,), np.nan, dtype=np.float64)

    for t in range(cfg.horizon - 1, -1, -1):
        for load in range(mdp.n_loads):
            for queue in range(cfg.max_queue + 1):
                for budget in range(cfg.max_budget + 1):
                    for action in range(mdp.n_actions):
                        q_values[t, load, queue, budget, action] = _action_value(
                            mdp, values, t, load, queue, budget, action
                        )
                    feasible_q = q_values[t, load, queue, budget]
                    if not np.isfinite(feasible_q).any():
                        raise RuntimeError("state has no feasible action")
                    best_action = int(np.nanargmax(feasible_q))
                    actions[t, load, queue, budget] = best_action
                    values[t, load, queue, budget] = feasible_q[best_action]

    a0 = int(np.argmin(mdp.action_costs))
    advantages = q_values - q_values[..., a0, None]
    residual = 0.0
    for index in np.ndindex(actions.shape):
        best = float(np.nanmax(q_values[index]))
        residual = max(residual, abs(float(values[index]) - best))
    return ActionDPResult(values, actions, q_values, advantages, residual)


def action_truth_frame(
    mdp: ActionConditionedBudgetMDP,
    result: ActionDPResult,
    scenario: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cfg = mdp.config
    for t, load, queue, budget, action in np.ndindex(result.q_values.shape):
        q_star = result.q_values[t, load, queue, budget, action]
        advantage = result.advantages[t, load, queue, budget, action]
        rows.append(
            {
                "scenario": scenario or cfg.scenario,
                "state": f"load={load};queue={queue}",
                "t": t,
                "load": load,
                "queue": queue,
                "remaining_budget": budget,
                "remaining_horizon": cfg.horizon - t,
                "action": action,
                "action_cost": int(mdp.action_costs[action]),
                "feasible": bool(np.isfinite(q_star)),
                "Q_star": float(q_star) if np.isfinite(q_star) else np.nan,
                "A_star": float(advantage) if np.isfinite(advantage) else np.nan,
                "optimal_action": int(result.actions[t, load, queue, budget]),
            }
        )
    return pd.DataFrame(rows)
