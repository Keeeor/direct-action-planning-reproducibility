"""Runtime-consistent Direct Action Planner with an independent safety input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .prototype_api import ACTION_ORDER, ActionMapper
from .transition import (
    RuntimeBranchPrediction,
    RuntimeConsistentSystemModel,
    target_is_feasible,
)


def action_invariant_forecast(checkpoint: Any, observation: np.ndarray) -> float:
    """Return the single causal forecast shared by every action branch.

    The v1 repair checkpoint used the neural point forecast directly.  The
    validation-system pilot showed that both learned forecasters had collapsed
    toward a profile mean and under-reacted to causal load measurements.  A
    checkpoint may therefore register the conservative ``causal_envelope``
    adapter.  It still produces exactly one action-independent forecast, but it
    cannot be lower than the current arrival observation or its causal EWMA.
    Legacy checkpoints default to their original ``learned`` behavior.
    """

    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (14,) or not np.isfinite(values).all():
        raise ValueError("planner observation must be a finite 14-vector")
    metadata = getattr(checkpoint, "metadata", {}) or {}
    strategy = str(
        getattr(checkpoint, "forecast_strategy", metadata.get("forecast_strategy", "learned"))
    )
    multiplier = float(
        getattr(checkpoint, "forecast_multiplier", metadata.get("forecast_multiplier", 1.0))
    )
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("forecast_multiplier must be finite and positive")
    learned = max(float(checkpoint.forecaster.predict(values)), 0.0)
    if strategy == "learned":
        forecast = learned
    elif strategy == "causal_envelope":
        forecast = max(learned, float(values[0]), float(values[1]), 0.0)
    else:
        raise ValueError(f"unknown forecast strategy: {strategy}")
    return float(forecast * multiplier)


@dataclass(frozen=True)
class RuntimePlanningDecision:
    action: str
    greedy_action: str
    target_replicas: int
    q_values: dict[str, float]
    feasible: dict[str, bool]
    branches: dict[str, RuntimeBranchPrediction]
    predicted_load_rps: float
    model_ready_replicas: int
    safety_ready_replicas: int
    tie_retained_current_target: bool


class RuntimeConsistentPlanner:
    def __init__(
        self,
        *,
        checkpoint: Any,
        system_model: RuntimeConsistentSystemModel,
        mapper: ActionMapper,
        control_interval_seconds: float,
        horizon_steps: int,
        total_budget_seconds: float,
        tie_margin: float = 0.0,
    ):
        self.checkpoint = checkpoint
        self.system_model = system_model
        self.mapper = mapper
        self.control_interval_seconds = float(control_interval_seconds)
        self.horizon_steps = int(horizon_steps)
        self.total_budget_seconds = float(total_budget_seconds)
        self.tie_margin = float(tie_margin)
        if self.tie_margin < 0 or not np.isfinite(self.tie_margin):
            raise ValueError("tie_margin must be finite and non-negative")

    def select(
        self,
        *,
        observation: np.ndarray,
        model_ready: int,
        safety_ready: int,
        remaining_budget_seconds: float,
        remaining_horizon_steps: int,
        current_target_replicas: int,
    ) -> RuntimePlanningDecision:
        forecast = action_invariant_forecast(self.checkpoint, observation)
        q_values: dict[str, float] = {}
        feasible: dict[str, bool] = {}
        branches: dict[str, RuntimeBranchPrediction] = {}
        for action in ACTION_ORDER:
            target = self.mapper.replicas(action)
            allowed = target_is_feasible(
                remaining_budget_seconds=remaining_budget_seconds,
                target_replicas=target,
                current_ready=safety_ready,
                base_replicas=self.system_model.base_replicas,
                control_interval_seconds=self.control_interval_seconds,
                scale_down_guard_seconds=self.system_model.scale_down_guard_seconds,
            )
            feasible[action] = allowed
            if not allowed:
                q_values[action] = float("-inf")
                continue
            branch = self.system_model.branch(
                observation=np.asarray(observation, dtype=np.float32),
                current_ready=int(model_ready), action=action, mapper=self.mapper,
                forecast_arrival_rps=forecast,
                total_budget_seconds=self.total_budget_seconds,
                remaining_budget_seconds=remaining_budget_seconds,
                remaining_horizon_steps=remaining_horizon_steps,
                horizon_steps=self.horizon_steps,
                control_interval_seconds=self.control_interval_seconds,
            )
            continuation = 0.0
            if remaining_horizon_steps > 1 and self.checkpoint.continuation_weight > 0:
                continuation = float(
                    self.checkpoint.value.predict(branch.next_observation.reshape(1, -1))[0]
                )
            branches[action] = branch
            q_values[action] = float(
                branch.reward
                + self.checkpoint.gamma
                * self.checkpoint.continuation_weight
                * continuation
            )
        if not any(feasible.values()):
            raise AssertionError("base action must always remain feasible")
        greedy = max(ACTION_ORDER, key=lambda name: q_values[name])
        selected = greedy
        retained = False
        try:
            current_action = ACTION_ORDER[
                self.mapper.action_index(int(current_target_replicas))
            ]
        except KeyError:
            current_action = ""
        if (
            current_action
            and feasible.get(current_action, False)
            and q_values[greedy] - q_values[current_action] <= self.tie_margin + 1.0e-12
        ):
            selected = current_action
            retained = selected != greedy
        return RuntimePlanningDecision(
            action=selected,
            greedy_action=greedy,
            target_replicas=self.mapper.replicas(selected),
            q_values=q_values,
            feasible=feasible,
            branches=branches,
            predicted_load_rps=forecast,
            model_ready_replicas=int(model_ready),
            safety_ready_replicas=int(safety_ready),
            tie_retained_current_target=retained,
        )
