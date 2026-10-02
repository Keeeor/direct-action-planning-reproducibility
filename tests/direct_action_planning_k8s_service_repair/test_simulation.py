from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from dap.direct_action_planning_k8s_service_repair.planner import (
    RuntimeConsistentPlanner,
)
from dap.direct_action_planning_k8s_service_repair.prototype_api import (
    ActionMapper,
)
from dap.direct_action_planning_k8s_service_repair.simulation import (
    evaluate_window,
)
from dap.direct_action_planning_k8s_service_repair.transition import (
    RuntimeConsistentSystemModel,
)


class ZeroForecaster:
    def predict(self, observation: np.ndarray) -> float:
        del observation
        return 0.0


class ZeroValue:
    def predict(self, observations: np.ndarray) -> np.ndarray:
        return np.zeros(len(observations), dtype=np.float32)


@dataclass
class Checkpoint:
    forecaster: ZeroForecaster
    value: ZeroValue
    gamma: float = 0.98
    continuation_weight: float = 1.0


def _components() -> tuple[RuntimeConsistentSystemModel, ActionMapper, RuntimeConsistentPlanner]:
    model = RuntimeConsistentSystemModel(
        profile="gold", capacity_by_replicas={1: 10.0, 2: 20.0, 3: 30.0, 5: 50.0},
        startup_delay_seconds=0.0, scale_down_guard_seconds=1.0, queue_max=100.0,
        reward={"completion_weight": 1.0, "queue_penalty": 0.002, "latency_penalty": 0.1, "slo_penalty": 1.0},
        slo_seconds=1.0,
    )
    mapper = ActionMapper({"no_op": 1, "scale_small": 2, "scale_medium": 3, "scale_large": 5})
    planner = RuntimeConsistentPlanner(
        checkpoint=Checkpoint(ZeroForecaster(), ZeroValue()), system_model=model, mapper=mapper,
        control_interval_seconds=5.0, horizon_steps=4,
        total_budget_seconds=100.0,
    )
    return model, mapper, planner


def test_closed_loop_environment_uses_true_arrivals_not_controller_forecast() -> None:
    model, mapper, planner = _components()
    metrics = evaluate_window(
        rates=np.asarray([0.0, 30.0, 30.0, 0.0]), planner=planner,
        system_model=model, mapper=mapper, total_budget_seconds=100.0,
        control_interval_seconds=5.0,
    )
    assert metrics.total_arrivals == 300.0
    assert max(float(row["next_queue"]) for row in metrics.steps) > 0.0
    assert all(row["predicted_load_rps"] == 0.0 for row in metrics.steps)
    assert any(row["true_arrival_rps"] == 30.0 for row in metrics.steps)


def test_fixed_window_evaluation_is_reproducible_and_finite() -> None:
    model, mapper, planner = _components()
    rates = np.asarray([4.0, 8.0, 12.0, 4.0])
    first = evaluate_window(
        rates=rates, planner=planner, system_model=model, mapper=mapper,
        total_budget_seconds=100.0, control_interval_seconds=5.0,
    )
    second = evaluate_window(
        rates=rates, planner=planner, system_model=model, mapper=mapper,
        total_budget_seconds=100.0, control_interval_seconds=5.0,
    )
    assert first.as_dict() == second.as_dict()
    assert np.isfinite(list(first.as_dict().values())[:-1]).all()
