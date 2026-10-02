from __future__ import annotations

from dap.direct_action_planning_k8s_service_repair.runtime import (
    decision_ready_inputs,
)
from dap.direct_action_planning_k8s_service_repair.collector import (
    stale_system_with_current_safety,
)
from tests.direct_action_planning_k8s_service_repair.test_collector import (
    _snapshot,
)


def test_runtime_decision_uses_stale_model_ready_but_current_safety_ready() -> None:
    stale = _snapshot(rate=8.0, completed=6.0, capacity=10.0, ready=1)
    current = _snapshot(rate=12.0, completed=10.0, capacity=30.0, ready=3)
    delivered = stale_system_with_current_safety(stale, current)
    model_ready, safety_ready = decision_ready_inputs(delivered)
    assert model_ready == 1
    assert safety_ready == 3

