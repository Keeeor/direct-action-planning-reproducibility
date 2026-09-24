from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class ActionSpec:
    name: str
    cost: float
    capacity_delta: float


DEFAULT_ACTIONS = (
    ActionSpec("no_op", 0.0, 0.0),
    ActionSpec("scale_small", 1.0, 1.0),
    ActionSpec("scale_medium", 2.0, 2.2),
    ActionSpec("scale_large", 4.0, 4.8),
)


@dataclass(frozen=True)
class SyntheticQueueConfig:
    horizon: int = 128
    budget: float = 128.0
    scenario: str = "stable"
    base_arrival_rate: float = 6.0
    base_capacity: float = 5.0
    slo_latency: float = 3.0
    rolling_window: int = 8
    max_queue: float = 500.0
    reward_completion: float = 1.0
    reward_queue_penalty: float = 0.04
    reward_latency_penalty: float = 0.08
    reward_slo_penalty: float = 2.0
    burst_scale: float = 2.4
    actions: tuple[ActionSpec, ...] = field(default_factory=lambda: DEFAULT_ACTIONS)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        if self.base_capacity <= 0:
            raise ValueError("base_capacity must be positive")
        if not self.actions or min(a.cost for a in self.actions) != 0.0:
            raise ValueError("at least one zero-cost action is required")
        if any(a.cost < 0 or a.capacity_delta < 0 for a in self.actions):
            raise ValueError("action costs and capacity deltas must be non-negative")


OBSERVATION_FIELDS = (
    "current_load",
    "recent_load_mean",
    "queue_length",
    "queue_growth",
    "active_capacity",
    "hot_resource_ratio",
    "utilization",
    "recent_mean_latency",
    "recent_tail_latency",
    "recent_slo_violation_rate",
    "burst_intensity",
    "resource_pressure",
    "remaining_budget_ratio",
    "remaining_horizon_ratio",
)


class DynamicBudgetSchedulingEnv(gym.Env[np.ndarray, int]):
    """Finite-horizon non-stationary queue with an episodic resource budget.

    An action changes service capacity for the current step only. This keeps the
    inter-temporal coupling attributable to the finite global budget, which is
    the mechanism under study. Arrival rates for future steps are stored by the
    simulator but never included in observations.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: SyntheticQueueConfig | None = None):
        super().__init__()
        self.config = config or SyntheticQueueConfig()
        self.action_costs = np.asarray([a.cost for a in self.config.actions], dtype=np.float64)
        self.capacity_deltas = np.asarray(
            [a.capacity_delta for a in self.config.actions], dtype=np.float64
        )
        self.action_space = spaces.Discrete(len(self.config.actions))
        self.observation_space = spaces.Box(
            low=np.full(len(OBSERVATION_FIELDS), -np.inf, dtype=np.float32),
            high=np.full(len(OBSERVATION_FIELDS), np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self._arrival_rates = np.empty(0, dtype=np.float64)
        self._arrivals = np.empty(0, dtype=np.float64)
        self._history_arrivals: deque[float] = deque(maxlen=self.config.rolling_window)
        self._history_latency: deque[float] = deque(maxlen=self.config.rolling_window)
        self._history_tail: deque[float] = deque(maxlen=self.config.rolling_window)
        self._history_slo: deque[float] = deque(maxlen=self.config.rolling_window)
        self._reset_state()

    def _reset_state(self) -> None:
        self.t = 0
        self.queue = 0.0
        self.previous_queue = 0.0
        self.previous_capacity = self.config.base_capacity
        self.previous_utilization = 0.0
        self.cumulative_cost = 0.0
        self.exhaustion_step: int | None = None
        self._history_arrivals.clear()
        self._history_latency.clear()
        self._history_tail.clear()
        self._history_slo.clear()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_state()
        if options and "arrival_trace" in options:
            trace = np.asarray(options["arrival_trace"], dtype=np.float64)
            if trace.shape != (self.config.horizon,):
                raise ValueError("arrival_trace length must equal horizon")
            if np.any(trace < 0) or not np.all(np.isfinite(trace)):
                raise ValueError("arrival_trace must contain finite non-negative values")
            self._arrival_rates = trace.copy()
            self._arrivals = trace.copy()
        else:
            self._arrival_rates = generate_load_profile(self.config)
            self._arrivals = self.np_random.poisson(self._arrival_rates).astype(np.float64)
        return self._observation(), {"observation_fields": OBSERVATION_FIELDS}

    def _observation(self) -> np.ndarray:
        if self.t >= self.config.horizon:
            current_load = 0.0
        else:
            current_load = float(self._arrival_rates[self.t])
        recent_load = (
            float(np.mean(self._history_arrivals))
            if self._history_arrivals
            else current_load
        )
        queue_growth = self.queue - self.previous_queue
        recent_mean_latency = float(np.mean(self._history_latency)) if self._history_latency else 1.0
        recent_tail = float(np.mean(self._history_tail)) if self._history_tail else 1.0
        recent_slo = float(np.mean(self._history_slo)) if self._history_slo else 0.0
        burst = current_load / max(recent_load, 1e-6)
        # Resource pressure is deliberately budget-independent. Otherwise B2/B3
        # would receive a disguised remaining-budget signal in their base state.
        resource_pressure = current_load / max(self.previous_capacity, 1e-8)
        remaining_ratio = (
            max(self.config.budget - self.cumulative_cost, 0.0) / self.config.budget
            if self.config.budget > 0
            else 0.0
        )
        remaining_horizon = max(self.config.horizon - self.t, 0) / self.config.horizon
        hot_ratio = min(max(self.previous_capacity - self.config.base_capacity, 0.0) / 4.8, 1.0)
        obs = np.asarray(
            [
                current_load,
                recent_load,
                self.queue,
                queue_growth,
                self.previous_capacity,
                hot_ratio,
                self.previous_utilization,
                recent_mean_latency,
                recent_tail,
                recent_slo,
                burst,
                resource_pressure,
                remaining_ratio,
                remaining_horizon,
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: int):
        if self.t >= self.config.horizon:
            raise RuntimeError("step called after episode termination")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        action = int(action)
        arrivals = float(self._arrivals[self.t])
        prior_queue = self.queue
        available = prior_queue + arrivals
        capacity = self.config.base_capacity + float(self.capacity_deltas[action])
        served = min(available, capacity)
        self.queue = min(max(available - served, 0.0), self.config.max_queue)
        utilization = served / max(capacity, 1e-8)
        mean_latency = 1.0 + (prior_queue + 0.5 * self.queue) / max(capacity, 1e-8)
        tail_latency = 1.0 + 2.0 * self.queue / max(capacity, 1e-8)
        slo_violation = int(tail_latency > self.config.slo_latency)
        completion_ratio = served / max(available, 1.0)
        reward = (
            self.config.reward_completion * completion_ratio
            - self.config.reward_queue_penalty * self.queue
            - self.config.reward_latency_penalty * tail_latency
            - self.config.reward_slo_penalty * slo_violation
        )

        step_cost = float(self.action_costs[action])
        self.cumulative_cost += step_cost
        if (
            self.exhaustion_step is None
            and self.config.budget > 0
            and self.cumulative_cost >= self.config.budget
        ):
            self.exhaustion_step = self.t + 1

        self._history_arrivals.append(arrivals)
        self._history_latency.append(mean_latency)
        self._history_tail.append(tail_latency)
        self._history_slo.append(float(slo_violation))
        self.previous_queue = prior_queue
        self.previous_capacity = capacity
        self.previous_utilization = utilization
        risk = self._risk_level(current_load=float(self._arrival_rates[self.t]))
        self.t += 1
        terminated = self.t >= self.config.horizon
        obs = self._observation()
        remaining = max(self.config.budget - self.cumulative_cost, 0.0)
        info = {
            "arrivals": arrivals,
            "served": served,
            "resource_cost": step_cost,
            "cumulative_cost": self.cumulative_cost,
            "remaining_budget": remaining,
            "remaining_budget_ratio": (
                remaining / self.config.budget if self.config.budget > 0 else 0.0
            ),
            "remaining_horizon_ratio": max(self.config.horizon - self.t, 0)
            / self.config.horizon,
            "slo_violation": slo_violation,
            "queue_length": self.queue,
            "mean_latency": mean_latency,
            "tail_latency": tail_latency,
            "load_level": float(self._arrival_rates[self.t - 1]),
            "risk_level": risk,
            "action_name": self.config.actions[action].name,
            "action": action,
            "capacity": capacity,
            "utilization": utilization,
            "budget_utilization": (
                self.cumulative_cost / self.config.budget if self.config.budget > 0 else 0.0
            ),
            "budget_exhaustion_ratio": (
                self.exhaustion_step / self.config.horizon
                if self.exhaustion_step is not None
                else 1.0
            ),
        }
        return obs, float(reward), terminated, False, info

    def _risk_level(self, current_load: float) -> float:
        load_pressure = current_load / max(self.config.base_capacity, 1e-8)
        queue_pressure = self.queue / max(self.config.base_capacity * 4.0, 1e-8)
        tail = self._history_tail[-1] / max(self.config.slo_latency, 1e-8)
        raw = 0.35 * load_pressure + 0.45 * queue_pressure + 0.20 * tail
        return float(1.0 - math.exp(-max(raw, 0.0)))


def generate_load_profile(config: SyntheticQueueConfig) -> np.ndarray:
    h = config.horizon
    t = np.arange(h, dtype=np.float64)
    base = config.base_arrival_rate
    scenario = config.scenario
    if scenario == "stable":
        profile = np.full(h, base)
    elif scenario == "periodic":
        profile = base * (0.75 + 0.35 * (1.0 + np.sin(2 * np.pi * t / max(h / 3, 4))))
    elif scenario in {"early_burst", "late_burst", "single_burst"}:
        center = 0.25 * h if scenario == "early_burst" else 0.75 * h
        width = max(h * 0.08, 1.0)
        profile = base * (0.75 + config.burst_scale * np.exp(-0.5 * ((t - center) / width) ** 2))
    elif scenario == "multi_burst":
        profile = np.full(h, base * 0.7)
        for center in (0.2 * h, 0.5 * h, 0.8 * h):
            width = max(h * 0.045, 1.0)
            profile += base * 1.55 * np.exp(-0.5 * ((t - center) / width) ** 2)
    elif scenario == "gradual":
        profile = base * (0.55 + 1.05 * t / max(h - 1, 1))
    elif scenario == "abrupt":
        profile = np.where(t < 0.55 * h, base * 0.65, base * 1.65)
    elif scenario == "ood_burst":
        center, width = 0.7 * h, max(h * 0.055, 1.0)
        profile = base * (0.65 + 3.25 * np.exp(-0.5 * ((t - center) / width) ** 2))
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return np.asarray(np.maximum(profile, 0.0), dtype=np.float64)
