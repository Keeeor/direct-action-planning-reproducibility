from __future__ import annotations

from dataclasses import replace

from dap.direct_action_planning_k8s_robustness.perturbations import (
    PerturbedCollector,
    dropout_steps,
)
from dap.direct_action_planning_k8s_robustness.prototype_api import (
    FieldEvidence,
    StateSnapshot,
)


def snapshot(step: int, budget: float, horizon: float) -> StateSnapshot:
    fields = {
        f"field_{index}": FieldEvidence(
            raw=float(step * 100 + index), normalized=float(step * 100 + index),
            timestamp=f"t{step}", freshness_seconds=0.0, missing_policy="test",
        )
        for index in range(12)
    }
    fields["remaining_budget_ratio"] = FieldEvidence(
        raw=budget, normalized=budget, timestamp=f"t{step}",
        freshness_seconds=0.0, missing_policy="current",
    )
    fields["remaining_horizon_ratio"] = replace(
        fields["remaining_budget_ratio"], raw=horizon, normalized=horizon
    )
    return StateSnapshot(
        collected_at=f"t{step}", monotonic_seconds=float(step),
        collection_latency_seconds=0.1, pod_metric_latency_seconds=0.05,
        desired_replicas=step + 1, ready_replicas=step + 1,
        missing_pods=(), fields=fields,
        dap_observation=tuple([float(step)] * 12 + [budget, horizon]),
    )


class FakeCollector:
    def __init__(self):
        self.step = 0

    def collect(self, *, remaining_budget_ratio: float, remaining_horizon_ratio: float):
        value = snapshot(self.step, remaining_budget_ratio, remaining_horizon_ratio)
        self.step += 1
        return value


def test_dropout_schedule_is_exact_reproducible_and_excludes_warmup():
    first = dropout_steps(horizon=64, fraction=0.10, seed=20260901)
    second = dropout_steps(horizon=64, fraction=0.10, seed=20260901)
    assert first == second
    assert len(first) == 6
    assert 0 not in first


def test_one_cycle_lag_preserves_current_budget_and_horizon_only():
    collector = PerturbedCollector(
        FakeCollector(), condition="observation_lag_1", horizon=64,
        seed=1, event_path=None,
    )
    first = collector.collect(remaining_budget_ratio=1.0, remaining_horizon_ratio=1.0)
    second = collector.collect(remaining_budget_ratio=0.8, remaining_horizon_ratio=0.5)
    assert first.dap_observation[:12] == tuple([0.0] * 12)
    assert second.dap_observation[:12] == tuple([0.0] * 12)
    assert second.dap_observation[12:] == (0.8, 0.5)
    assert second.ready_replicas == first.ready_replicas
    assert second.fields["remaining_budget_ratio"].raw == 0.8
    assert second.fields["remaining_horizon_ratio"].raw == 0.5


def test_dropout_reuses_last_delivered_system_observation():
    collector = PerturbedCollector(
        FakeCollector(), condition="metric_dropout_10pct", horizon=10,
        seed=5, event_path=None, forced_dropout_steps={1},
    )
    first = collector.collect(remaining_budget_ratio=1.0, remaining_horizon_ratio=1.0)
    second = collector.collect(remaining_budget_ratio=0.9, remaining_horizon_ratio=0.9)
    third = collector.collect(remaining_budget_ratio=0.8, remaining_horizon_ratio=0.8)
    assert second.dap_observation[:12] == first.dap_observation[:12]
    assert second.dap_observation[12:] == (0.9, 0.9)
    assert third.dap_observation[:12] == tuple([2.0] * 12)
    assert collector.applied_events == 1
