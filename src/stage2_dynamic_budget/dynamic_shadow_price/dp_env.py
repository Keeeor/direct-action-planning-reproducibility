from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dp_reference import DiscreteBudgetMDP, DiscreteDPConfig


class DiscreteBudgetGymEnv(gym.Env[np.ndarray, int]):
    """Gym adapter whose transition and reward exactly match the DP reference."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: DiscreteDPConfig,
        initial_budget: int,
        budget_scale: int | None = None,
    ):
        super().__init__()
        if not 0 <= initial_budget <= config.max_budget:
            raise ValueError("initial_budget outside DP grid")
        self.config = config
        self.initial_budget = int(initial_budget)
        self.budget_scale = int(budget_scale or initial_budget)
        if self.budget_scale < self.initial_budget or self.budget_scale <= 0:
            raise ValueError("budget_scale must be positive and at least initial_budget")
        self.mdp = DiscreteBudgetMDP(config)
        self.action_space = spaces.Discrete(len(config.action_costs))
        self.observation_space = spaces.Box(
            low=np.full(14, -np.inf, dtype=np.float32),
            high=np.full(14, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self._reset_state()

    def _reset_state(self) -> None:
        self.t = 0
        self.load = 1
        self.queue = 0
        self.previous_queue = 0
        self.previous_capacity = 1
        self.previous_utilization = 0.0
        self.previous_violation = 0.0
        self.remaining_budget = self.initial_budget
        self.cumulative_cost = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("options are not supported")
        self._reset_state()
        return self._observation(), {}

    def valid_action_mask(self) -> np.ndarray:
        mask = self.mdp.action_costs <= self.remaining_budget
        mask[int(np.argmin(self.mdp.action_costs))] = True
        return mask.astype(bool)

    def _observation(self) -> np.ndarray:
        arrivals = float(self.mdp.load_arrivals[self.load])
        queue_growth = float(self.queue - self.previous_queue)
        latency = 1.0 + self.queue / max(self.previous_capacity, 1)
        tail = 1.0 + 2.0 * self.queue / max(self.previous_capacity, 1)
        budget_ratio = self.remaining_budget / self.budget_scale
        horizon_ratio = max(self.config.horizon - self.t, 0) / self.config.horizon
        return np.asarray(
            [
                arrivals,
                arrivals,
                float(self.queue),
                queue_growth,
                float(self.previous_capacity),
                float(max(self.previous_capacity - 1, 0) / 3),
                self.previous_utilization,
                latency,
                tail,
                self.previous_violation,
                arrivals / 2.0,
                arrivals / max(self.previous_capacity, 1),
                budget_ratio,
                horizon_ratio,
            ],
            dtype=np.float32,
        )

    def step(self, action: int):
        if self.t >= self.config.horizon:
            raise RuntimeError("step called after termination")
        if not self.action_space.contains(action):
            raise ValueError("invalid action")
        action = int(action)
        cost = int(self.mdp.action_costs[action])
        if cost > self.remaining_budget:
            raise ValueError("action cost exceeds remaining budget")
        prior_queue = self.queue
        next_queue, reward, metrics = self.mdp.outcome(self.queue, self.load, action)
        probabilities = self.mdp.load_probabilities(self.t, self.load)
        next_load = int(self.np_random.choice(self.mdp.n_loads, p=probabilities))
        self.remaining_budget -= cost
        self.cumulative_cost += cost
        self.previous_queue = prior_queue
        self.previous_capacity = int(self.mdp.action_capacity[action])
        self.previous_utilization = metrics["served"] / max(self.previous_capacity, 1)
        self.previous_violation = metrics["slo_violation"]
        self.queue = next_queue
        current_load = self.load
        self.load = next_load
        self.t += 1
        terminated = self.t >= self.config.horizon
        info = {
            "arrivals": float(self.mdp.load_arrivals[current_load]),
            "served": metrics["served"],
            "resource_cost": float(cost),
            "cumulative_cost": float(self.cumulative_cost),
            "remaining_budget": float(self.remaining_budget),
            "remaining_budget_ratio": self.remaining_budget / self.budget_scale,
            "remaining_horizon_ratio": max(self.config.horizon - self.t, 0)
            / self.config.horizon,
            "slo_violation": metrics["slo_violation"],
            "queue_length": float(self.queue),
            "mean_latency": 1.0 + 0.5 * self.queue,
            "tail_latency": 1.0 + float(self.queue),
            "load_level": float(self.mdp.load_arrivals[current_load]),
            "risk_level": float(current_load + prior_queue),
            "action": action,
            "capacity": float(self.previous_capacity),
            "utilization": float(self.previous_utilization),
            "budget_utilization": (
                self.cumulative_cost / self.initial_budget if self.initial_budget > 0 else 0.0
            ),
        }
        return self._observation(), float(reward), terminated, False, info
