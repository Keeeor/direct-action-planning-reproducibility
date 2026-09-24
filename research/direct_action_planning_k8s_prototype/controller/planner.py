from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .action_mapper import ACTION_ORDER, ActionMapper
from .budget_tracker import BudgetTracker
from .checkpoint_loader import DAPCheckpoint
from .system_model import BranchPrediction, StructuredSystemModel


@dataclass(frozen=True)
class PlanningDecision:
    action: str
    target_replicas: int
    q_values: dict[str, float]
    feasible: dict[str, bool]
    branches: dict[str, BranchPrediction]
    predicted_load_rps: float


class DirectActionPlanner:
    def __init__(
        self, *, checkpoint: DAPCheckpoint, system_model: StructuredSystemModel,
        mapper: ActionMapper, control_interval_seconds: float, horizon_steps: int,
        total_budget_seconds: float,
    ):
        self.checkpoint = checkpoint
        self.system_model = system_model
        self.mapper = mapper
        self.control_interval_seconds = float(control_interval_seconds)
        self.horizon_steps = int(horizon_steps)
        self.total_budget_seconds = float(total_budget_seconds)

    def select(
        self, *, observation: np.ndarray, state: Mapping[str, float], budget: BudgetTracker,
        current_ready: int, remaining_horizon_steps: int,
    ) -> PlanningDecision:
        forecast = max(float(self.checkpoint.forecaster.predict(observation)), 0.0)
        q_values: dict[str, float] = {}
        feasible: dict[str, bool] = {}
        branches: dict[str, BranchPrediction] = {}
        for action in ACTION_ORDER:
            target = self.mapper.replicas(action)
            allowed = budget.can_target(
                target, current_ready=current_ready,
                control_interval=self.control_interval_seconds,
                scale_down_guard_seconds=self.system_model.scale_down_guard_seconds,
            )
            feasible[action] = allowed
            if not allowed:
                q_values[action] = float("-inf")
                continue
            branch = self.system_model.branch(
                observation=observation, state=state, action=action, mapper=self.mapper,
                forecast_arrival_rps=forecast,
                total_budget_seconds=self.total_budget_seconds, remaining_budget_seconds=budget.remaining,
                remaining_horizon_steps=remaining_horizon_steps, horizon_steps=self.horizon_steps,
                control_interval_seconds=self.control_interval_seconds,
            )
            continuation = 0.0
            if remaining_horizon_steps > 1 and self.checkpoint.continuation_weight > 0:
                continuation = float(self.checkpoint.value.predict(branch.next_observation.reshape(1, -1))[0])
            branches[action] = branch
            q_values[action] = float(
                branch.reward + self.checkpoint.gamma * self.checkpoint.continuation_weight * continuation
            )
        action = max(ACTION_ORDER, key=lambda name: q_values[name])
        if not feasible[action]:
            raise AssertionError("no hard-budget-feasible action available")
        return PlanningDecision(
            action=action, target_replicas=self.mapper.replicas(action), q_values=q_values,
            feasible=feasible, branches=branches, predicted_load_rps=forecast,
        )
