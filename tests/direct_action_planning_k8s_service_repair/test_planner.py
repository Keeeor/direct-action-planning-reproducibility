from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from dap.direct_action_planning_k8s_service_repair.planner import (
    RuntimeConsistentPlanner,
)
from dap.direct_action_planning_k8s_service_repair.prototype_api import (
    ActionMapper,
)
from dap.direct_action_planning_k8s_service_repair.transition import (
    RuntimeConsistentSystemModel,
)


class FixedForecaster:
    def __init__(self, prediction: float):
        self.prediction = prediction

    def predict(self, observation: np.ndarray) -> float:
        del observation
        return self.prediction


class CapacityValue:
    def predict(self, observations: np.ndarray) -> np.ndarray:
        return np.asarray(observations)[:, 4] / 10.0


class TerminalValueMustNotBeCalled:
    def predict(self, observations: np.ndarray) -> np.ndarray:
        del observations
        raise AssertionError("terminal continuation value was evaluated")


@dataclass
class Checkpoint:
    forecaster: FixedForecaster
    value: CapacityValue
    gamma: float = 1.0
    continuation_weight: float = 1.0
    forecast_strategy: str = "learned"
    forecast_multiplier: float = 1.0


def _planner(tie_margin: float = 0.0, guard: float = 0.0) -> RuntimeConsistentPlanner:
    model = RuntimeConsistentSystemModel(
        profile="gold", capacity_by_replicas={1: 10.0, 2: 20.0, 3: 30.0, 5: 50.0},
        startup_delay_seconds=0.0, scale_down_guard_seconds=guard, queue_max=100.0,
        reward={"completion_weight": 0.0, "queue_penalty": 0.0, "latency_penalty": 0.0, "slo_penalty": 0.0},
        slo_seconds=1.0,
    )
    return RuntimeConsistentPlanner(
        checkpoint=Checkpoint(FixedForecaster(7.0), CapacityValue()),
        system_model=model,
        mapper=ActionMapper({"no_op": 1, "scale_small": 2, "scale_medium": 3, "scale_large": 5}),
        control_interval_seconds=5.0, horizon_steps=4,
        total_budget_seconds=100.0, tie_margin=tie_margin,
    )


def _observation() -> np.ndarray:
    return np.asarray(
        [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        dtype=np.float32,
    )


def test_controller_forecast_is_shared_by_all_action_branches() -> None:
    decision = _planner().select(
        observation=_observation(), model_ready=1, safety_ready=1,
        remaining_budget_seconds=100.0, remaining_horizon_steps=4,
        current_target_replicas=1,
    )
    assert decision.predicted_load_rps == 7.0
    assert set(branch.details["forecast_arrival_rps"] for branch in decision.branches.values()) == {7.0}
    assert decision.action == "scale_large"


def test_causal_envelope_repairs_collapsed_point_forecast_and_remains_shared() -> None:
    planner = _planner()
    planner.checkpoint.forecast_strategy = "causal_envelope"
    planner.checkpoint.forecast_multiplier = 1.1
    observation = _observation()
    observation[0] = 13.0
    observation[1] = 11.0
    decision = planner.select(
        observation=observation, model_ready=1, safety_ready=1,
        remaining_budget_seconds=100.0, remaining_horizon_steps=4,
        current_target_replicas=1,
    )
    assert np.isclose(decision.predicted_load_rps, 14.3)
    assert set(
        branch.details["forecast_arrival_rps"]
        for branch in decision.branches.values()
    ) == {decision.predicted_load_rps}


def test_legacy_checkpoint_retains_learned_forecast_semantics() -> None:
    decision = _planner().select(
        observation=_observation(), model_ready=1, safety_ready=1,
        remaining_budget_seconds=100.0, remaining_horizon_steps=4,
        current_target_replicas=1,
    )
    assert decision.predicted_load_rps == 7.0


def test_true_next_load_is_not_a_planner_argument() -> None:
    parameters = RuntimeConsistentPlanner.select.__annotations__
    assert "true_next_load" not in parameters


def test_current_target_tie_rule_is_explicit_and_bounded() -> None:
    planner = _planner(tie_margin=3.1)
    decision = planner.select(
        observation=_observation(), model_ready=1, safety_ready=1,
        remaining_budget_seconds=100.0, remaining_horizon_steps=4,
        current_target_replicas=2,
    )
    assert decision.greedy_action == "scale_large"
    assert decision.action == "scale_small"
    assert decision.tie_retained_current_target


def test_safety_ready_not_model_ready_controls_feasibility() -> None:
    planner = _planner(guard=3.0)
    decision = planner.select(
        observation=_observation(), model_ready=1, safety_ready=5,
        remaining_budget_seconds=20.0, remaining_horizon_steps=4,
        current_target_replicas=1,
    )
    # A stale learned Ready=1 admits target 3 at this budget, while the current
    # safety Ready=5 requires a larger scale-down reserve and rejects it.
    assert decision.safety_ready_replicas == 5
    assert decision.model_ready_replicas == 1
    assert not decision.feasible["scale_medium"]


def test_last_kubernetes_decision_uses_immediate_reward_only() -> None:
    planner = _planner()
    planner.checkpoint.value = TerminalValueMustNotBeCalled()
    decision = planner.select(
        observation=_observation(), model_ready=1, safety_ready=1,
        remaining_budget_seconds=100.0, remaining_horizon_steps=1,
        current_target_replicas=1,
    )
    for action, branch in decision.branches.items():
        assert decision.q_values[action] == branch.reward
