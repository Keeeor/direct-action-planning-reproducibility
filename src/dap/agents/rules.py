from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class NoIntervention:
    name = "b0_no_intervention"

    def act(self, observation: np.ndarray, **_: object) -> int:
        return 0


@dataclass(frozen=True)
class ThresholdRule:
    name: str
    thresholds: tuple[float, float, float]

    def risk_score(self, observation: np.ndarray) -> float:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape[-1] < 14:
            raise ValueError("expected the canonical 14-field observation")
        load, queue, growth = obs[0], obs[2], max(obs[3], 0.0)
        tail_latency, recent_slo = obs[8], obs[9]
        return float(
            queue
            + 1.5 * growth
            + 2.0 * max(load - 5.0, 0.0)
            + 3.0 * max(tail_latency - 1.0, 0.0)
            + 8.0 * recent_slo
        )

    def act(self, observation: np.ndarray, **_: object) -> int:
        score = self.risk_score(observation)
        low, medium, high = self.thresholds
        if score >= high:
            return 3
        if score >= medium:
            return 2
        if score >= low:
            return 1
        return 0


class ConservativeRule(ThresholdRule):
    def __init__(self):
        super().__init__("b1_conservative", (8.0, 20.0, 40.0))


class AggressiveRule(ThresholdRule):
    def __init__(self):
        super().__init__("b1_aggressive", (3.0, 12.0, 25.0))
