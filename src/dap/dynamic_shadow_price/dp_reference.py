from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DiscreteDPConfig:
    horizon: int = 16
    max_budget: int = 12
    max_queue: int = 6
    scenario: str = "early_burst"
    gamma: float = 0.99
    action_costs: tuple[int, ...] = (0, 1, 2, 3)
    action_capacity: tuple[int, ...] = (1, 2, 3, 4)
    load_arrivals: tuple[int, ...] = (1, 2, 4)
    queue_penalty: float = 0.45
    severe_queue_penalty: float = 1.5

    def __post_init__(self) -> None:
        if self.horizon <= 1 or self.max_budget < 0 or self.max_queue <= 0:
            raise ValueError("invalid horizon, budget, or queue range")
        if self.scenario not in {"stable", "early_burst", "late_burst"}:
            raise ValueError("unknown DP scenario")
        if len(self.action_costs) != 4 or len(self.action_capacity) != 4:
            raise ValueError("the DP reference requires four actions")
        if self.action_costs[0] != 0 or any(cost < 0 for cost in self.action_costs):
            raise ValueError("the first action must be zero-cost and all costs non-negative")


class DiscreteBudgetMDP:
    """Small time-inhomogeneous queue MDP with an exact finite-budget solution."""

    def __init__(self, config: DiscreteDPConfig):
        self.config = config
        self.action_costs = np.asarray(config.action_costs, dtype=np.int64)
        self.action_capacity = np.asarray(config.action_capacity, dtype=np.int64)
        self.load_arrivals = np.asarray(config.load_arrivals, dtype=np.int64)
        self.n_loads = len(config.load_arrivals)

    def load_probabilities(self, t: int, current_load: int) -> np.ndarray:
        if not 0 <= current_load < self.n_loads:
            raise ValueError("invalid load state")
        persistence = np.full(self.n_loads, 0.1, dtype=np.float64)
        persistence[current_load] = 0.8
        center = 0.25 if self.config.scenario == "early_burst" else 0.75
        phase = t / max(self.config.horizon - 1, 1)
        burst = np.exp(-0.5 * ((phase - center) / 0.11) ** 2)
        if self.config.scenario == "stable":
            target = np.asarray([0.15, 0.70, 0.15])
            mix = 0.25
        else:
            target = np.asarray([0.05, 0.15, 0.80])
            mix = 0.65 * burst
        probabilities = (1.0 - mix) * persistence + mix * target
        return probabilities / probabilities.sum()

    def outcome(self, queue: int, load: int, action: int) -> tuple[int, float, dict[str, float]]:
        arrivals = int(self.load_arrivals[load])
        available = queue + arrivals
        served = min(available, int(self.action_capacity[action]))
        next_queue = min(max(available - served, 0), self.config.max_queue)
        violation = float(next_queue >= max(3, self.config.max_queue // 2))
        reward = (
            float(served)
            - self.config.queue_penalty * float(next_queue)
            - self.config.severe_queue_penalty * violation
        )
        metrics = {
            "served": float(served),
            "queue": float(next_queue),
            "slo_violation": violation,
            "cost": float(self.action_costs[action]),
        }
        return next_queue, reward, metrics

    def transitions(
        self, t: int, load: int, queue: int, action: int
    ) -> Iterable[tuple[float, int, int, float]]:
        next_queue, reward, _ = self.outcome(queue, load, action)
        for next_load, probability in enumerate(self.load_probabilities(t, load)):
            if probability > 0:
                yield float(probability), next_load, next_queue, reward


@dataclass(frozen=True)
class DPResult:
    values: np.ndarray
    actions: np.ndarray
    shadow_prices: np.ndarray
    max_bellman_residual: float


def solve_backward_dp(mdp: DiscreteBudgetMDP) -> DPResult:
    cfg = mdp.config
    values = np.zeros(
        (cfg.horizon + 1, mdp.n_loads, cfg.max_queue + 1, cfg.max_budget + 1),
        dtype=np.float64,
    )
    actions = np.zeros(
        (cfg.horizon, mdp.n_loads, cfg.max_queue + 1, cfg.max_budget + 1),
        dtype=np.int64,
    )
    residual = 0.0
    for t in range(cfg.horizon - 1, -1, -1):
        for load in range(mdp.n_loads):
            for queue in range(cfg.max_queue + 1):
                for budget in range(cfg.max_budget + 1):
                    candidates: list[float] = []
                    feasible: list[int] = []
                    for action, cost in enumerate(mdp.action_costs):
                        if cost > budget:
                            continue
                        expected = 0.0
                        for probability, next_load, next_queue, reward in mdp.transitions(
                            t, load, queue, action
                        ):
                            expected += probability * (
                                reward
                                + cfg.gamma
                                * values[t + 1, next_load, next_queue, budget - int(cost)]
                            )
                        candidates.append(expected)
                        feasible.append(action)
                    best_index = int(np.argmax(candidates))
                    best_action = feasible[best_index]
                    best_value = candidates[best_index]
                    actions[t, load, queue, budget] = best_action
                    values[t, load, queue, budget] = best_value
                    residual = max(residual, abs(best_value - max(candidates)))
    shadow = np.full_like(values, np.nan)
    shadow[..., 1:] = values[..., 1:] - values[..., :-1]
    return DPResult(values, actions, shadow, residual)


def simulate_optimal_policy(
    mdp: DiscreteBudgetMDP,
    result: DPResult,
    initial_budget: int,
    seed: int,
    initial_load: int = 1,
    initial_queue: int = 0,
) -> list[dict[str, float | int]]:
    if not 0 <= initial_budget <= mdp.config.max_budget:
        raise ValueError("initial_budget is outside the solved budget grid")
    rng = np.random.default_rng(seed)
    load = int(initial_load)
    queue = int(initial_queue)
    budget = int(initial_budget)
    rows: list[dict[str, float | int]] = []
    cumulative_reward = 0.0
    cumulative_cost = 0
    for t in range(mdp.config.horizon):
        action = int(result.actions[t, load, queue, budget])
        cost = int(mdp.action_costs[action])
        if cost > budget:
            raise RuntimeError("DP policy selected an infeasible action")
        next_queue, reward, metrics = mdp.outcome(queue, load, action)
        probabilities = mdp.load_probabilities(t, load)
        next_load = int(rng.choice(mdp.n_loads, p=probabilities))
        budget -= cost
        cumulative_cost += cost
        cumulative_reward += reward
        rows.append(
            {
                "t": t,
                "remaining_horizon": mdp.config.horizon - t,
                "load": load,
                "queue": queue,
                "risk_score": float(load + queue),
                "budget_before": budget + cost,
                "action": action,
                "cost": cost,
                "reward": float(reward),
                "served": float(metrics["served"]),
                "slo_violation": float(metrics["slo_violation"]),
                "next_load": next_load,
                "next_queue": next_queue,
                "remaining_budget": budget,
                "shadow_price": (
                    float(result.shadow_prices[t, load, queue, budget + cost])
                    if budget + cost > 0
                    else None
                ),
                "cumulative_cost": cumulative_cost,
                "cumulative_reward": float(cumulative_reward),
            }
        )
        load, queue = next_load, next_queue
    return rows
