"""One source of truth for DAP branch mechanics and hard feasibility."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .prototype_api import ACTION_ORDER, ActionMapper


def conservative_commitment_seconds(
    *, target_replicas: int, current_ready: int, base_replicas: int,
    control_interval_seconds: float, scale_down_guard_seconds: float,
) -> float:
    """Worst-case new commitment used identically in every execution phase."""

    target = int(target_replicas)
    current = int(current_ready)
    base = int(base_replicas)
    if target < base or current < 0 or base < 1:
        return float("inf")
    if target == base:
        return 0.0
    current_extra = max(current - base, 0)
    target_extra = max(target - base, 0)
    return float(
        target_extra * max(float(control_interval_seconds), 0.0)
        + max(current_extra, target_extra) * max(float(scale_down_guard_seconds), 0.0)
    )


def target_is_feasible(
    *, remaining_budget_seconds: float, target_replicas: int, current_ready: int,
    base_replicas: int, control_interval_seconds: float,
    scale_down_guard_seconds: float,
) -> bool:
    if int(target_replicas) == int(base_replicas):
        return True
    commitment = conservative_commitment_seconds(
        target_replicas=target_replicas, current_ready=current_ready,
        base_replicas=base_replicas,
        control_interval_seconds=control_interval_seconds,
        scale_down_guard_seconds=scale_down_guard_seconds,
    )
    return commitment <= float(remaining_budget_seconds) + 1.0e-9


@dataclass(frozen=True)
class RuntimeBranchPrediction:
    action: str
    target_replicas: int
    effective_replicas: float
    next_ready_replicas: int
    expected_cost_seconds: float
    reward: float
    next_observation: np.ndarray
    details: dict[str, float]


@dataclass(frozen=True)
class RuntimeConsistentSystemModel:
    profile: str
    capacity_by_replicas: Mapping[int, float]
    startup_delay_seconds: float
    scale_down_guard_seconds: float
    queue_max: float
    reward: Mapping[str, float]
    base_replicas: int = 1
    slo_seconds: float = 1.0

    @classmethod
    def load(
        cls, path: str | Path, profile: str, *, slo_seconds: float
    ) -> "RuntimeConsistentSystemModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "dap.k8s.structured_system_model.v1":
            raise ValueError("unrecognized system-model schema")
        spec = payload.get("profiles", {}).get(profile)
        if not spec:
            raise ValueError(f"system model lacks profile {profile!r}")
        capacities = {
            int(key): float(value)
            for key, value in spec["capacity_by_ready_replicas_rps"].items()
        }
        if 1 not in capacities or any(value <= 0 for value in capacities.values()):
            raise ValueError("capacity curve must contain positive base capacity")
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
        values = np.asarray(
            [self.capacity_by_replicas[int(key)] for key in keys], dtype=np.float64
        )
        return float(np.interp(float(replicas), keys, values))

    def effective_replicas(
        self, current_ready: int, target_replicas: int,
        control_interval_seconds: float,
    ) -> float:
        interval = max(float(control_interval_seconds), 1.0e-9)
        current = max(int(current_ready), self.base_replicas)
        target = max(int(target_replicas), self.base_replicas)
        if target <= current:
            lag = min(self.scale_down_guard_seconds / interval, 1.0)
            return float(current - (current - target) * (1.0 - lag))
        available_fraction = max(interval - self.startup_delay_seconds, 0.0) / interval
        return float(current + (target - current) * available_fraction)

    def next_ready_replicas(
        self, current_ready: int, target_replicas: int,
        control_interval_seconds: float,
    ) -> int:
        interval = max(float(control_interval_seconds), 0.0)
        current = max(int(current_ready), self.base_replicas)
        target = max(int(target_replicas), self.base_replicas)
        if target > current and self.startup_delay_seconds > interval:
            return current
        if target < current and self.scale_down_guard_seconds > interval:
            return current
        return target

    def branch(
        self,
        *,
        observation: np.ndarray,
        current_ready: int,
        action: str,
        mapper: ActionMapper,
        forecast_arrival_rps: float,
        total_budget_seconds: float,
        remaining_budget_seconds: float,
        remaining_horizon_steps: int,
        horizon_steps: int,
        control_interval_seconds: float,
    ) -> RuntimeBranchPrediction:
        if action not in ACTION_ORDER:
            raise KeyError(action)
        target = mapper.replicas(action)
        effective = self.effective_replicas(
            current_ready, target, control_interval_seconds
        )
        next_ready = self.next_ready_replicas(
            current_ready, target, control_interval_seconds
        )
        interval_capacity = self.capacity(effective)
        next_capacity = self.capacity(next_ready)
        queue = max(float(observation[2]), 0.0)
        arrival = max(float(forecast_arrival_rps), 0.0)
        interval = max(float(control_interval_seconds), 1.0e-9)
        arrivals = arrival * interval
        service = min(queue + arrivals, interval_capacity * interval)
        next_queue = max(queue + arrivals - service, 0.0)
        expected_cost = max(effective - self.base_replicas, 0.0) * interval
        interval_utilization = service / max(interval_capacity * interval, 1.0e-9)
        # Live collection divides completed request rate over the just-finished
        # interval by capacity at the next decision's current Ready state.
        service_utilization = (service / interval) / max(next_capacity, 1.0e-9)
        mean_latency = max(1.0 / max(interval_capacity, 1.0e-9), 1.0e-4) + (
            queue + 0.5 * next_queue
        ) / max(interval_capacity, 1.0e-9)
        tail_latency = mean_latency + next_queue / max(interval_capacity, 1.0e-9)
        slo = float(tail_latency > self.slo_seconds)
        completion = service / max(queue + arrivals, 1.0)
        reward = (
            self.reward["completion_weight"] * completion
            - self.reward["queue_penalty"] * next_queue / max(self.queue_max, 1.0)
            - self.reward["latency_penalty"] * tail_latency / max(self.slo_seconds, 1.0e-9)
            - self.reward["slo_penalty"] * slo
            - 0.05 * expected_cost / max(float(total_budget_seconds), 1.0)
        )
        remaining = max(float(remaining_budget_seconds) - expected_cost, 0.0)
        recent = float(observation[1])
        next_recent = 0.75 * recent + 0.25 * arrival
        burst = arrival / max(next_recent, 1.0e-9) if next_recent > 0 else 0.0
        pressure = (arrival + next_queue / interval) / max(next_capacity, 1.0e-9)
        max_target = max(mapper.targets.values())
        next_observation = np.asarray(
            [
                arrival,
                next_recent,
                next_queue,
                (next_queue - queue) / interval,
                next_capacity,
                max(next_ready - self.base_replicas, 0)
                / max(max_target - self.base_replicas, 1),
                service_utilization,
                mean_latency,
                tail_latency,
                slo,
                burst,
                pressure,
                remaining / max(float(total_budget_seconds), 1.0),
                max(int(remaining_horizon_steps) - 1, 0) / max(int(horizon_steps), 1),
            ],
            dtype=np.float32,
        )
        return RuntimeBranchPrediction(
            action=action,
            target_replicas=target,
            effective_replicas=effective,
            next_ready_replicas=next_ready,
            expected_cost_seconds=float(expected_cost),
            reward=float(reward),
            next_observation=next_observation,
            details={
                "forecast_arrival_rps": arrival,
                "interval_capacity_rps": interval_capacity,
                "next_ready_capacity_rps": next_capacity,
                "served_requests": service,
                "arrivals_plus_queue": queue + arrivals,
                "next_queue": next_queue,
                "tail_latency_seconds": tail_latency,
                "slo_violation": slo,
                "completion_ratio": completion,
                "interval_utilization": interval_utilization,
                "service_utilization": service_utilization,
            },
        )

