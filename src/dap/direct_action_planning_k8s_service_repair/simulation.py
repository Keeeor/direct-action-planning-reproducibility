"""Closed-loop validation that mirrors the runtime planning path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .planner import RuntimeConsistentPlanner
from .prototype_api import ACTION_ORDER, ActionMapper
from .transition import RuntimeConsistentSystemModel, target_is_feasible


@dataclass(frozen=True)
class SimulationMetrics:
    completion_ratio: float
    slo_violation_rate: float
    ready_cost_seconds: float
    total_reward: float
    total_arrivals: float
    total_served: float
    final_queue: float
    target_changes: int
    budget_violation_seconds: float
    steps: tuple[dict[str, float | int | str | bool], ...]

    def as_dict(self) -> dict:
        return asdict(self)


def initial_observation(
    model: RuntimeConsistentSystemModel, rate: float,
    total_budget_seconds: float, horizon_steps: int,
) -> np.ndarray:
    capacity = model.capacity(model.base_replicas)
    return np.asarray(
        [
            float(rate), float(rate), 0.0, 0.0, capacity, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            float(rate) / max(capacity, 1.0e-9),
            float(total_budget_seconds > 0), 1.0,
        ],
        dtype=np.float32,
    )


Policy = Callable[[np.ndarray, int, float, int, int], str]


def evaluate_window(
    *,
    rates: np.ndarray,
    system_model: RuntimeConsistentSystemModel,
    mapper: ActionMapper,
    total_budget_seconds: float,
    control_interval_seconds: float,
    planner: RuntimeConsistentPlanner | None = None,
    policy: Policy | None = None,
) -> SimulationMetrics:
    """Evaluate a controller without leaking true next load into action scores."""

    values = np.asarray(rates, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("rates must be a finite one-dimensional window")
    if (planner is None) == (policy is None):
        raise ValueError("provide exactly one of planner or policy")
    horizon = len(values)
    observation = initial_observation(
        system_model, float(values[0]), total_budget_seconds, horizon
    )
    ready = system_model.base_replicas
    current_target = ready
    remaining = float(total_budget_seconds)
    total_arrivals = total_served = total_slo_served = 0.0
    total_reward = total_cost = 0.0
    target_changes = 0
    rows: list[dict[str, float | int | str | bool]] = []
    for step in range(horizon):
        remaining_horizon = horizon - step
        true_arrival = float(values[step + 1]) if step + 1 < horizon else 0.0
        if planner is not None:
            decision = planner.select(
                observation=observation,
                model_ready=ready,
                safety_ready=ready,
                remaining_budget_seconds=remaining,
                remaining_horizon_steps=remaining_horizon,
                current_target_replicas=current_target,
            )
            action = decision.action
            predicted_load = decision.predicted_load_rps
            greedy_action = decision.greedy_action
            tie_retained = decision.tie_retained_current_target
        else:
            action = str(policy(observation, ready, remaining, remaining_horizon, current_target))
            if action not in ACTION_ORDER:
                raise ValueError(f"policy returned invalid action: {action}")
            target = mapper.replicas(action)
            if not target_is_feasible(
                remaining_budget_seconds=remaining, target_replicas=target,
                current_ready=ready, base_replicas=system_model.base_replicas,
                control_interval_seconds=control_interval_seconds,
                scale_down_guard_seconds=system_model.scale_down_guard_seconds,
            ):
                action = "no_op"
            predicted_load = float("nan")
            greedy_action = action
            tie_retained = False
        actual = system_model.branch(
            observation=observation, current_ready=ready, action=action,
            mapper=mapper, forecast_arrival_rps=true_arrival,
            total_budget_seconds=total_budget_seconds,
            remaining_budget_seconds=remaining,
            remaining_horizon_steps=remaining_horizon,
            horizon_steps=horizon,
            control_interval_seconds=control_interval_seconds,
        )
        total_arrivals += true_arrival * control_interval_seconds
        total_served += actual.details["served_requests"]
        total_slo_served += (
            actual.details["served_requests"] * actual.details["slo_violation"]
        )
        total_cost += actual.expected_cost_seconds
        total_reward += actual.reward
        remaining -= actual.expected_cost_seconds
        if actual.target_replicas != current_target:
            target_changes += 1
        rows.append(
            {
                "step": step,
                "action": action,
                "greedy_action": greedy_action,
                "target_replicas": actual.target_replicas,
                "current_ready_replicas": ready,
                "next_ready_replicas": actual.next_ready_replicas,
                "predicted_load_rps": predicted_load,
                "true_arrival_rps": true_arrival,
                "served_requests": actual.details["served_requests"],
                "next_queue": actual.details["next_queue"],
                "ready_cost_seconds": actual.expected_cost_seconds,
                "tie_retained_current_target": tie_retained,
            }
        )
        current_target = actual.target_replicas
        ready = actual.next_ready_replicas
        observation = actual.next_observation
    completion = min(total_served / max(total_arrivals, 1.0), 1.0)
    slo = total_slo_served / max(total_served, 1.0)
    return SimulationMetrics(
        completion_ratio=float(completion),
        slo_violation_rate=float(slo),
        ready_cost_seconds=float(total_cost),
        total_reward=float(total_reward),
        total_arrivals=float(total_arrivals),
        total_served=float(total_served),
        final_queue=float(observation[2]),
        target_changes=int(target_changes),
        budget_violation_seconds=max(float(total_cost) - float(total_budget_seconds), 0.0),
        steps=tuple(rows),
    )

