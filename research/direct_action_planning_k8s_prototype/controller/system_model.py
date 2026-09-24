from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .action_mapper import ACTION_ORDER, ActionMapper


@dataclass(frozen=True)
class BranchPrediction:
    action: str
    target_replicas: int
    effective_replicas: float
    expected_cost_seconds: float
    reward: float
    next_observation: np.ndarray
    details: dict[str, float]


@dataclass(frozen=True)
class StructuredSystemModel:
    """Known queue/action mechanics calibrated from the real Kubernetes service.

    This is deliberately not a learned full-state transition. Its only input
    forecast is the action-invariant next external arrival rate. Replica
    capacity, queue drain, latency response, budget update, and horizon update
    are calculated explicitly for every candidate action.
    """

    profile: str
    capacity_by_replicas: Mapping[int, float]
    startup_delay_seconds: float
    scale_down_guard_seconds: float
    queue_max: float
    reward: Mapping[str, float]
    base_replicas: int = 1
    slo_seconds: float = 1.0

    @classmethod
    def load(cls, path: Path, profile: str, *, slo_seconds: float) -> "StructuredSystemModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "dap.k8s.structured_system_model.v1":
            raise ValueError("unrecognized system-model schema")
        spec = payload.get("profiles", {}).get(profile)
        if not spec:
            raise ValueError(f"system model lacks profile {profile!r}")
        raw = spec["capacity_by_ready_replicas_rps"]
        capacities = {int(key): float(value) for key, value in raw.items()}
        if 1 not in capacities or any(value <= 0 for value in capacities.values()):
            raise ValueError("calibrated capacity must be positive for every replica level")
        return cls(
            profile=profile,
            capacity_by_replicas=capacities,
            startup_delay_seconds=float(spec["startup_delay_seconds_median"]),
            scale_down_guard_seconds=float(spec["startup_delay_seconds_guard"]),
            queue_max=max(float(spec.get("queue_max", 1.0)), 1.0),
            reward={key: float(value) for key, value in payload["reward"].items()},
            slo_seconds=float(slo_seconds),
        )

    def capacity(self, replicas: float) -> float:
        keys = np.asarray(sorted(self.capacity_by_replicas), dtype=np.float64)
        values = np.asarray([self.capacity_by_replicas[int(key)] for key in keys], dtype=np.float64)
        return float(np.interp(float(replicas), keys, values))

    def _effective_replicas(
        self, current_ready: int, target_replicas: int, control_interval_seconds: float
    ) -> float:
        if target_replicas <= current_ready:
            # Scaling down is observable only after actual Ready-Pod changes;
            # retain a conservative latency fraction in one-step planning.
            lag = min(self.scale_down_guard_seconds / max(control_interval_seconds, 1.0e-9), 1.0)
            return float(current_ready - (current_ready - target_replicas) * (1.0 - lag))
        available_fraction = max(
            control_interval_seconds - self.startup_delay_seconds, 0.0
        ) / max(control_interval_seconds, 1.0e-9)
        return float(current_ready + (target_replicas - current_ready) * available_fraction)

    def branch(
        self,
        *,
        observation: np.ndarray,
        state: Mapping[str, float],
        action: str,
        mapper: ActionMapper,
        forecast_arrival_rps: float,
        total_budget_seconds: float,
        remaining_budget_seconds: float,
        remaining_horizon_steps: int,
        horizon_steps: int,
        control_interval_seconds: float,
    ) -> BranchPrediction:
        if action not in ACTION_ORDER:
            raise KeyError(action)
        target = mapper.replicas(action)
        current_ready = max(int(round(state.get("ready_pods", self.base_replicas))), self.base_replicas)
        effective = self._effective_replicas(current_ready, target, control_interval_seconds)
        capacity_rps = self.capacity(effective)
        queue = max(float(state.get("queue_depth", 0.0)), 0.0)
        arrival = max(float(forecast_arrival_rps), 0.0)
        arrivals = arrival * control_interval_seconds
        service = min(queue + arrivals, capacity_rps * control_interval_seconds)
        next_queue = max(queue + arrivals - service, 0.0)
        extra = max(effective - self.base_replicas, 0.0)
        expected_cost = extra * control_interval_seconds
        utilization = service / max(capacity_rps * control_interval_seconds, 1.0e-9)
        mean_latency = max(1.0 / max(capacity_rps, 1.0e-9), 1.0e-4) + (
            queue + 0.5 * next_queue
        ) / max(capacity_rps, 1.0e-9)
        tail_latency = mean_latency + next_queue / max(capacity_rps, 1.0e-9)
        slo = float(tail_latency > self.slo_seconds)
        completion = service / max(queue + arrivals, 1.0)
        normalized_queue = next_queue / max(self.queue_max, 1.0)
        normalized_latency = tail_latency / max(self.slo_seconds, 1.0e-9)
        reward = (
            self.reward["completion_weight"] * completion
            - self.reward["queue_penalty"] * normalized_queue
            - self.reward["latency_penalty"] * normalized_latency
            - self.reward["slo_penalty"] * slo
            - 0.05 * expected_cost / max(total_budget_seconds, 1.0)
        )
        remaining = max(remaining_budget_seconds - expected_cost, 0.0)
        recent_load = float(observation[1])
        next_recent_load = 0.75 * recent_load + 0.25 * arrival
        burst = arrival / max(next_recent_load, 1.0e-9)
        pressure = (arrival + next_queue / max(control_interval_seconds, 1.0)) / max(capacity_rps, 1.0e-9)
        next_observation = np.asarray(
            [
                arrival,
                next_recent_load,
                next_queue,
                next_queue - queue,
                capacity_rps,
                max(effective - self.base_replicas, 0.0) / max(max(mapper.targets.values()) - self.base_replicas, 1),
                utilization,
                mean_latency,
                tail_latency,
                slo,
                burst,
                pressure,
                remaining / max(total_budget_seconds, 1.0),
                max(remaining_horizon_steps - 1, 0) / max(horizon_steps, 1),
            ],
            dtype=np.float32,
        )
        return BranchPrediction(
            action=action, target_replicas=target, effective_replicas=effective,
            expected_cost_seconds=expected_cost, reward=float(reward),
            next_observation=next_observation,
            details={
                "forecast_arrival_rps": arrival, "capacity_rps": capacity_rps,
                "served_requests": service, "next_queue": next_queue,
                "tail_latency_seconds": tail_latency, "slo_violation": slo,
                "completion_ratio": completion,
            },
        )
