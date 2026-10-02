from __future__ import annotations

import numpy as np

from dap.direct_action_planning_k8s_service_repair.transition import (
    RuntimeConsistentSystemModel,
    conservative_commitment_seconds,
    target_is_feasible,
)
from dap.direct_action_planning_k8s_service_repair.prototype_api import (
    ActionMapper,
)


def _model(*, startup: float = 2.0, guard: float = 3.0) -> RuntimeConsistentSystemModel:
    return RuntimeConsistentSystemModel(
        profile="gold",
        capacity_by_replicas={1: 10.0, 2: 20.0, 3: 30.0, 5: 50.0},
        startup_delay_seconds=startup,
        scale_down_guard_seconds=guard,
        queue_max=100.0,
        reward={
            "completion_weight": 1.0,
            "queue_penalty": 0.002,
            "latency_penalty": 0.1,
            "slo_penalty": 1.0,
        },
        slo_seconds=1.0,
    )


def _observation() -> np.ndarray:
    return np.asarray(
        [8.0, 8.0, 0.0, 0.0, 10.0, 0.0, 0.8, 0.1, 0.2, 0.0, 1.0, 0.8, 1.0, 1.0],
        dtype=np.float32,
    )


def test_interval_effective_and_next_decision_ready_are_distinct() -> None:
    model = _model(startup=2.5)
    mapper = ActionMapper({"no_op": 1, "scale_small": 2, "scale_medium": 3, "scale_large": 5})
    branch = model.branch(
        observation=_observation(),
        current_ready=1,
        action="scale_medium",
        mapper=mapper,
        forecast_arrival_rps=25.0,
        total_budget_seconds=100.0,
        remaining_budget_seconds=100.0,
        remaining_horizon_steps=4,
        horizon_steps=4,
        control_interval_seconds=5.0,
    )
    assert branch.effective_replicas == 2.0
    assert branch.next_ready_replicas == 3
    assert branch.next_observation[4] == 30.0
    assert branch.next_observation[5] == 0.5
    assert branch.details["interval_capacity_rps"] == 20.0


def test_next_ready_remains_current_if_startup_exceeds_interval() -> None:
    model = _model(startup=6.0)
    mapper = ActionMapper({"no_op": 1, "scale_small": 2, "scale_medium": 3, "scale_large": 5})
    branch = model.branch(
        observation=_observation(), current_ready=1, action="scale_large", mapper=mapper,
        forecast_arrival_rps=20.0, total_budget_seconds=100.0,
        remaining_budget_seconds=100.0, remaining_horizon_steps=4,
        horizon_steps=4, control_interval_seconds=5.0,
    )
    assert branch.effective_replicas == 1.0
    assert branch.next_ready_replicas == 1
    assert branch.next_observation[4] == 10.0


def test_one_conservative_feasibility_formula_covers_current_and_target_ready() -> None:
    assert conservative_commitment_seconds(
        target_replicas=3, current_ready=2, base_replicas=1,
        control_interval_seconds=5.0, scale_down_guard_seconds=3.0,
    ) == 16.0
    assert not target_is_feasible(
        remaining_budget_seconds=15.999, target_replicas=3, current_ready=2,
        base_replicas=1, control_interval_seconds=5.0,
        scale_down_guard_seconds=3.0,
    )
    assert target_is_feasible(
        remaining_budget_seconds=16.0, target_replicas=3, current_ready=2,
        base_replicas=1, control_interval_seconds=5.0,
        scale_down_guard_seconds=3.0,
    )
    assert target_is_feasible(
        remaining_budget_seconds=0.0, target_replicas=1, current_ready=5,
        base_replicas=1, control_interval_seconds=5.0,
        scale_down_guard_seconds=3.0,
    )


def test_budget_and_horizon_update_are_post_action() -> None:
    model = _model(startup=0.0)
    mapper = ActionMapper({"no_op": 1, "scale_small": 2, "scale_medium": 3, "scale_large": 5})
    branch = model.branch(
        observation=_observation(), current_ready=1, action="scale_small", mapper=mapper,
        forecast_arrival_rps=8.0, total_budget_seconds=100.0,
        remaining_budget_seconds=60.0, remaining_horizon_steps=4,
        horizon_steps=4, control_interval_seconds=5.0,
    )
    assert branch.expected_cost_seconds == 5.0
    assert np.isclose(branch.next_observation[12], 0.55)
    assert np.isclose(branch.next_observation[13], 0.75)

