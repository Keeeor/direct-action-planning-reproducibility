from __future__ import annotations

from dap.direct_action_planning_k8s_service_repair.perturbations import (
    PerturbedSemanticCollector,
    dropout_steps,
)
from tests.direct_action_planning_k8s_service_repair.test_collector import (
    SequenceCollector,
    _snapshot,
)


def test_dropout_schedule_is_deterministic_and_never_drops_first_step() -> None:
    first = dropout_steps(horizon=32, fraction=0.10, seed=7)
    second = dropout_steps(horizon=32, fraction=0.10, seed=7)
    assert first == second
    assert len(first) == 3
    assert 0 not in first


def test_lag_stales_learned_system_but_not_current_ready_safety() -> None:
    collector = PerturbedSemanticCollector(
        SequenceCollector([
            _snapshot(rate=8.0, completed=6.0, capacity=10.0, ready=1),
            _snapshot(rate=12.0, completed=10.0, capacity=30.0, ready=3),
        ]),
        condition="observation_lag_1", horizon=2, seed=7, event_path=None,
    )
    collector.collect(remaining_budget_ratio=1.0, remaining_horizon_ratio=1.0)
    delivered = collector.collect(
        remaining_budget_ratio=0.8, remaining_horizon_ratio=0.5
    )
    assert delivered.fields["ready_pods"].raw == 1.0
    assert delivered.ready_replicas == 3
    assert delivered.dap_observation[0] == 8.0
    assert delivered.dap_observation[12:] == (0.8, 0.7)
    assert collector.applied_events == 1
