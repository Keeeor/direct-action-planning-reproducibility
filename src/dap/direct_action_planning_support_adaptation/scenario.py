from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dap.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
)


@dataclass(frozen=True)
class ContinuousScenario:
    scenario_id: str
    base_load: float
    burst_start: float
    burst_amplitude: float
    burst_duration: float
    period: float
    periodic_amplitude: float

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id is required")
        if not 1.0 <= self.base_load <= 3.0:
            raise ValueError("base_load must be in [1, 3]")
        if not 0.0 <= self.burst_start <= 1.0:
            raise ValueError("burst_start must be in [0, 1]")
        if min(self.burst_amplitude, self.periodic_amplitude) < 0.0:
            raise ValueError("load amplitudes must be non-negative")
        if self.burst_duration <= 0.0 or self.period <= 0.0:
            raise ValueError("duration and period must be positive")

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "ContinuousScenario":
        return cls(
            scenario_id=str(payload["id"]),
            base_load=float(payload["base_load"]),
            burst_start=float(payload["burst_start"]),
            burst_amplitude=float(payload["burst_amplitude"]),
            burst_duration=float(payload["burst_duration"]),
            period=float(payload["period"]),
            periodic_amplitude=float(payload["periodic_amplitude"]),
        )

    def parameter_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.base_load,
                self.burst_start,
                self.burst_amplitude,
                self.burst_duration,
                self.period,
                self.periodic_amplitude,
            ],
            dtype=np.float64,
        )


class ContinuousLoadMDP(ActionConditionedBudgetMDP):
    """Exact queue MDP driven by a continuous six-parameter load process."""

    def __init__(self, config: ACBADPConfig, scenario_spec: ContinuousScenario):
        super().__init__(config)
        self.scenario_spec = scenario_spec
        arrivals = self.load_arrivals.astype(np.float64)
        scale = max(float(np.ptp(arrivals)) / 3.0, 0.5)
        weights = np.exp(-0.5 * ((arrivals - scenario_spec.base_load) / scale) ** 2)
        self._base_probabilities = weights / weights.sum()

    def load_probabilities(self, t: int, current_load: int) -> np.ndarray:
        if not 0 <= current_load < self.n_loads:
            raise ValueError("invalid load state")
        phase = float(t) / max(self.config.horizon - 1, 1)
        spec = self.scenario_spec
        center = spec.burst_start + 0.5 * spec.burst_duration
        sigma = max(spec.burst_duration / 2.355, 0.025)
        burst = spec.burst_amplitude * np.exp(-0.5 * ((phase - center) / sigma) ** 2)
        periodic = spec.periodic_amplitude * 0.5 * (
            1.0 + np.sin(2.0 * np.pi * phase / spec.period - np.pi / 2.0)
        )
        event_strength = float(np.clip(burst + periodic, 0.0, 0.90))
        persistent = np.full(self.n_loads, 0.10, dtype=np.float64)
        persistent[current_load] = 0.80
        background = 0.55 * persistent + 0.45 * self._base_probabilities
        high_load = np.asarray([0.04, 0.14, 0.82], dtype=np.float64)
        probabilities = (1.0 - event_strength) * background + event_strength * high_load
        probabilities = np.maximum(probabilities, 1.0e-8)
        return probabilities / probabilities.sum()


def build_continuous_mdp(
    scenario: ContinuousScenario,
    *,
    horizon: int = 16,
    max_budget: int = 12,
    max_queue: int = 6,
    gamma: float = 0.99,
    action_costs: tuple[int, ...] = (0, 1, 2, 3),
    action_capacity: tuple[int, ...] = (1, 2, 3, 4),
) -> ContinuousLoadMDP:
    # The parent config uses a valid canonical label; the exact continuous ID is
    # stored separately and never passed as a learned feature.
    config = ACBADPConfig(
        horizon=horizon,
        max_budget=max_budget,
        max_queue=max_queue,
        scenario="stable",
        gamma=gamma,
        action_costs=action_costs,
        action_capacity=action_capacity,
    )
    return ContinuousLoadMDP(config, scenario)
