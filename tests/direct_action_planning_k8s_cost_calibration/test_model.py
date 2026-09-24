from __future__ import annotations

import numpy as np
import pytest

from stage2_dynamic_budget.direct_action_planning_k8s_cost_calibration.model import (
    CostWeightedSystemModel,
)
from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.prototype_api import (
    ActionMapper,
)
from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.transition import (
    RuntimeConsistentSystemModel,
)


def _base_model() -> RuntimeConsistentSystemModel:
    return RuntimeConsistentSystemModel(
        profile="gentd_inference",
        capacity_by_replicas={1: 12.0, 2: 23.0, 3: 33.0, 5: 50.0},
        startup_delay_seconds=1.0,
        scale_down_guard_seconds=2.0,
        queue_max=500.0,
        reward={
            "completion_weight": 1.0,
            "queue_penalty": 0.4,
            "latency_penalty": 0.3,
            "slo_penalty": 0.5,
        },
        slo_seconds=1.0,
    )


def _branch(model: object, action: str):
    mapper = ActionMapper({
        "no_op": 1,
        "scale_small": 2,
        "scale_medium": 3,
        "scale_large": 5,
    })
    observation = np.zeros(14, dtype=np.float32)
    observation[1] = 8.0
    observation[12] = 1.0
    observation[13] = 1.0
    return model.branch(
        observation=observation,
        current_ready=1,
        action=action,
        mapper=mapper,
        forecast_arrival_rps=8.0,
        total_budget_seconds=256.0,
        remaining_budget_seconds=256.0,
        remaining_horizon_steps=32,
        horizon_steps=32,
        control_interval_seconds=5.0,
    )


def test_legacy_weight_reproduces_branch_exactly() -> None:
    base = _base_model()
    legacy = _branch(base, "scale_medium")
    calibrated = _branch(CostWeightedSystemModel(base, cost_weight=0.05), "scale_medium")
    assert calibrated.reward == pytest.approx(legacy.reward, abs=1e-12)
    assert calibrated.expected_cost_seconds == legacy.expected_cost_seconds
    np.testing.assert_array_equal(calibrated.next_observation, legacy.next_observation)
    assert calibrated.details == legacy.details


def test_cost_weight_only_changes_existing_cost_term() -> None:
    base = _base_model()
    low = _branch(CostWeightedSystemModel(base, cost_weight=0.10), "scale_medium")
    high = _branch(CostWeightedSystemModel(base, cost_weight=0.40), "scale_medium")
    expected = -(0.40 - 0.10) * low.expected_cost_seconds / 256.0
    assert high.reward - low.reward == pytest.approx(expected, abs=1e-12)
    assert high.target_replicas == low.target_replicas
    assert high.effective_replicas == low.effective_replicas
    assert high.next_ready_replicas == low.next_ready_replicas
    np.testing.assert_array_equal(high.next_observation, low.next_observation)


def test_cost_weight_has_no_effect_on_zero_cost_branch() -> None:
    base = _base_model()
    low = _branch(CostWeightedSystemModel(base, cost_weight=0.05), "no_op")
    high = _branch(CostWeightedSystemModel(base, cost_weight=0.80), "no_op")
    assert low.expected_cost_seconds == 0.0
    assert high.reward == pytest.approx(low.reward, abs=1e-12)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_cost_weight_must_be_finite_and_nonnegative(value: float) -> None:
    with pytest.raises(ValueError, match="cost_weight"):
        CostWeightedSystemModel(_base_model(), cost_weight=value)

