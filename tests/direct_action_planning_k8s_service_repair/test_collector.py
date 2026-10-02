from __future__ import annotations

from dataclasses import replace

from dap.direct_action_planning_k8s_service_repair.collector import (
    RuntimeSemanticCollector,
    stale_system_with_current_safety,
)
from dap.direct_action_planning_k8s_service_repair.prototype_api import (
    FieldEvidence,
    StateSnapshot,
)


def _field(value: float) -> FieldEvidence:
    return FieldEvidence(
        raw=value, normalized=value, timestamp="2026-08-09T00:00:00+00:00",
        freshness_seconds=0.0, missing_policy="gold",
    )


def _snapshot(*, rate: float, completed: float, capacity: float, ready: int) -> StateSnapshot:
    values = {
        "current_request_rate": rate,
        "recent_request_rate": rate,
        "queue_depth": 2.0,
        "queue_growth_rate": 0.5,
        "ready_pods": float(ready),
        "cpu_utilization": 0.9,
        "mean_latency_seconds": 0.1,
        "p95_latency_seconds": 0.2,
        "slo_violation_rate": 0.0,
        "burst_intensity": 1.0,
        "queue_to_capacity": 0.5,
        "remaining_budget_ratio": 0.8,
        "remaining_horizon_ratio": 0.7,
        "completed_request_rate": completed,
        "active_capacity_rps": capacity,
    }
    fields = {name: _field(value) for name, value in values.items()}
    observation = (
        rate, rate, 2.0, 0.5, capacity, (ready - 1) / 4.0, 0.9,
        0.1, 0.2, 0.0, 1.0, 0.5, 0.8, 0.7,
    )
    return StateSnapshot(
        collected_at="2026-08-09T00:00:00+00:00", monotonic_seconds=1.0,
        collection_latency_seconds=0.01, pod_metric_latency_seconds=0.01,
        desired_replicas=ready, ready_replicas=ready, missing_pods=(),
        fields=fields, dap_observation=observation,
    )


class SequenceCollector:
    def __init__(self, snapshots: list[StateSnapshot]):
        self.snapshots = list(snapshots)

    def collect(self, **_: float) -> StateSnapshot:
        return self.snapshots.pop(0)


def test_live_observation_uses_service_utilization_and_training_ewma() -> None:
    collector = RuntimeSemanticCollector(
        SequenceCollector([
            _snapshot(rate=8.0, completed=6.0, capacity=10.0, ready=1),
            _snapshot(rate=12.0, completed=10.0, capacity=20.0, ready=2),
        ])
    )
    first = collector.collect(remaining_budget_ratio=0.8, remaining_horizon_ratio=0.7)
    second = collector.collect(remaining_budget_ratio=0.6, remaining_horizon_ratio=0.5)
    assert first.dap_observation[1] == 8.0
    assert first.dap_observation[6] == 0.6
    assert second.dap_observation[1] == 9.0
    assert second.dap_observation[6] == 0.5
    assert second.fields["cpu_utilization"].raw == 0.9
    assert second.fields["service_utilization"].raw == 0.5


def test_stale_learned_system_preserves_current_top_level_ready_for_safety() -> None:
    stale = _snapshot(rate=8.0, completed=6.0, capacity=10.0, ready=1)
    current = replace(
        _snapshot(rate=12.0, completed=10.0, capacity=30.0, ready=3),
        fields={
            **_snapshot(rate=12.0, completed=10.0, capacity=30.0, ready=3).fields,
            "remaining_budget_ratio": _field(0.4),
            "remaining_horizon_ratio": _field(0.5),
        },
        dap_observation=tuple(
            _snapshot(rate=12.0, completed=10.0, capacity=30.0, ready=3).dap_observation[:12]
        ) + (0.4, 0.5),
    )
    delivered = stale_system_with_current_safety(stale, current)
    assert delivered.ready_replicas == 3
    assert delivered.desired_replicas == 3
    assert delivered.fields["ready_pods"].raw == 1.0
    assert delivered.dap_observation[:12] == stale.dap_observation[:12]
    assert delivered.dap_observation[12:] == (0.4, 0.5)

