from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from controller.action_mapper import ACTION_ORDER, ActionMapper
from controller.budget_tracker import BudgetTracker
from controller.checkpoint_loader import DAPCheckpoint
from controller.state_collector import StateSnapshot
from controller.system_model import BranchPrediction, StructuredSystemModel


@dataclass(frozen=True)
class BaselineDecision:
    action: str
    scores: dict[str, float]
    details: dict[str, float | int | str]


class ReplicaPolicy(Protocol):
    def select(
        self, *, snapshot: StateSnapshot, budget: BudgetTracker,
        remaining_horizon_steps: int,
    ) -> BaselineDecision: ...


def _safe_action(
    proposed: str, *, mapper: ActionMapper, budget: BudgetTracker,
    current_ready: int, control_interval_seconds: float, scale_down_guard_seconds: float,
) -> str:
    if budget.can_target(
        mapper.replicas(proposed), current_ready=current_ready,
        control_interval=control_interval_seconds, scale_down_guard_seconds=scale_down_guard_seconds,
    ):
        return proposed
    for action in reversed(ACTION_ORDER):
        if budget.can_target(
            mapper.replicas(action), current_ready=current_ready,
            control_interval=control_interval_seconds, scale_down_guard_seconds=scale_down_guard_seconds,
        ):
            return action
    return "no_op"


class StaticPolicy:
    def __init__(self, *, mapper: ActionMapper, replicas: int, control_interval_seconds: float, guard_seconds: float):
        self.mapper = mapper
        self.replicas = int(replicas)
        self.action = ACTION_ORDER[self.mapper.action_index(self.replicas)]
        self.interval = float(control_interval_seconds)
        self.guard = float(guard_seconds)

    def select(self, *, snapshot: StateSnapshot, budget: BudgetTracker, remaining_horizon_steps: int) -> BaselineDecision:
        del remaining_horizon_steps
        action = _safe_action(
            self.action, mapper=self.mapper, budget=budget, current_ready=snapshot.ready_replicas,
            control_interval_seconds=self.interval, scale_down_guard_seconds=self.guard,
        )
        return BaselineDecision(action=action, scores={name: float(name == action) for name in ACTION_ORDER}, details={"mode": "static"})


class ThresholdPolicy:
    def __init__(
        self, *, mapper: ActionMapper, control_interval_seconds: float, guard_seconds: float,
        slo_seconds: float, queue_small: float, queue_medium: float, queue_large: float,
    ):
        self.mapper = mapper
        self.interval = float(control_interval_seconds)
        self.guard = float(guard_seconds)
        self.slo = float(slo_seconds)
        self.thresholds = (float(queue_small), float(queue_medium), float(queue_large))

    def select(self, *, snapshot: StateSnapshot, budget: BudgetTracker, remaining_horizon_steps: int) -> BaselineDecision:
        del remaining_horizon_steps
        fields = snapshot.fields
        queue = fields["queue_depth"].raw
        p95 = fields["p95_latency_seconds"].raw
        if queue >= self.thresholds[2] or p95 >= self.slo * 1.5:
            proposed = "scale_large"
        elif queue >= self.thresholds[1] or p95 >= self.slo:
            proposed = "scale_medium"
        elif queue >= self.thresholds[0] or fields["queue_growth_rate"].raw > 0:
            proposed = "scale_small"
        else:
            proposed = "no_op"
        action = _safe_action(
            proposed, mapper=self.mapper, budget=budget, current_ready=snapshot.ready_replicas,
            control_interval_seconds=self.interval, scale_down_guard_seconds=self.guard,
        )
        return BaselineDecision(
            action=action, scores={name: float(name == proposed) for name in ACTION_ORDER},
            details={"mode": "threshold", "proposed_action": proposed, "queue": queue, "p95": p95},
        )


class MPCPolicy:
    def __init__(
        self, *, checkpoint: DAPCheckpoint, system_model: StructuredSystemModel, mapper: ActionMapper,
        total_budget_seconds: float, control_interval_seconds: float, horizon: int = 4,
    ):
        self.checkpoint = checkpoint
        self.model = system_model
        self.mapper = mapper
        self.total_budget = float(total_budget_seconds)
        self.interval = float(control_interval_seconds)
        self.horizon = int(horizon)

    @staticmethod
    def _raw_state(observation: np.ndarray, ready: int) -> dict[str, float]:
        return {
            "queue_depth": float(observation[2]), "ready_pods": float(ready),
            "current_request_rate": float(observation[0]), "p95_latency_seconds": float(observation[8]),
        }

    def _rollout(
        self, observation: np.ndarray, ready: int, remaining: float, remaining_horizon_steps: int,
        depth: int,
    ) -> float:
        if depth <= 0 or remaining_horizon_steps <= 0:
            return 0.0
        forecast = max(float(self.checkpoint.forecaster.predict(observation)), 0.0)
        best = -np.inf
        for action in ACTION_ORDER:
            target = self.mapper.replicas(action)
            extra = max(target - 1, 0)
            reserve = extra * self.model.scale_down_guard_seconds
            if action != "no_op" and extra * self.interval + reserve > remaining + 1.0e-9:
                continue
            branch = self.model.branch(
                observation=observation, state=self._raw_state(observation, ready), action=action,
                mapper=self.mapper, forecast_arrival_rps=forecast,
                total_budget_seconds=self.total_budget, remaining_budget_seconds=remaining,
                remaining_horizon_steps=remaining_horizon_steps,
                horizon_steps=max(remaining_horizon_steps, 1), control_interval_seconds=self.interval,
            )
            score = branch.reward + self.checkpoint.gamma * self._rollout(
                branch.next_observation, target, max(remaining - branch.expected_cost_seconds, 0.0),
                remaining_horizon_steps - 1, depth - 1,
            )
            best = max(best, score)
        return float(best if np.isfinite(best) else 0.0)

    def select(self, *, snapshot: StateSnapshot, budget: BudgetTracker, remaining_horizon_steps: int) -> BaselineDecision:
        observation = np.asarray(snapshot.dap_observation, dtype=np.float32)
        forecast = max(float(self.checkpoint.forecaster.predict(observation)), 0.0)
        scores: dict[str, float] = {}
        for action in ACTION_ORDER:
            target = self.mapper.replicas(action)
            if not budget.can_target(
                target, current_ready=snapshot.ready_replicas, control_interval=self.interval,
                scale_down_guard_seconds=self.model.scale_down_guard_seconds,
            ):
                scores[action] = -np.inf
                continue
            branch = self.model.branch(
                observation=observation, state=self._raw_state(observation, snapshot.ready_replicas),
                action=action, mapper=self.mapper, forecast_arrival_rps=forecast,
                total_budget_seconds=self.total_budget, remaining_budget_seconds=budget.remaining,
                remaining_horizon_steps=remaining_horizon_steps, horizon_steps=remaining_horizon_steps,
                control_interval_seconds=self.interval,
            )
            scores[action] = branch.reward + self.checkpoint.gamma * self._rollout(
                branch.next_observation, target,
                max(budget.remaining - branch.expected_cost_seconds, 0.0),
                remaining_horizon_steps - 1, self.horizon - 1,
            )
        action = max(ACTION_ORDER, key=lambda name: scores[name])
        return BaselineDecision(
            action=action, scores=scores,
            details={"mode": "mpc", "horizon": self.horizon, "predicted_load_rps": forecast},
        )
