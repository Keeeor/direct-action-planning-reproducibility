from controller.action_mapper import ACTION_ORDER, ActionMapper
from controller.budget_tracker import BudgetTracker
from baselines.autoscaler_runtime import native_budget_cap


def test_action_mapping_is_absolute_not_incremental() -> None:
    mapper = ActionMapper(dict(zip(ACTION_ORDER, (1, 2, 3, 5))))
    assert mapper.replicas("no_op") == 1
    assert mapper.replicas("scale_medium") == 3
    assert mapper.action_index(5) == 3


def test_ready_replica_second_integral_uses_previous_observed_state() -> None:
    tracker = BudgetTracker(100.0, base_replicas=1)
    tracker.update(ready_replicas=1, desired_replicas=3, monotonic_seconds=0.0)
    tracker.update(ready_replicas=3, desired_replicas=3, monotonic_seconds=2.0)
    tracker.update(ready_replicas=3, desired_replicas=1, monotonic_seconds=7.0)
    tracker.update(ready_replicas=1, desired_replicas=1, monotonic_seconds=9.0)
    assert tracker.ready_cost == 14.0
    assert tracker.requested_cost == 14.0


def test_budget_mask_reserves_scale_down_delay() -> None:
    tracker = BudgetTracker(22.0)
    tracker.update(ready_replicas=1, desired_replicas=1, monotonic_seconds=0.0)
    assert tracker.can_target(2, current_ready=1, control_interval=10, scale_down_guard_seconds=2)
    assert not tracker.can_target(3, current_ready=1, control_interval=10, scale_down_guard_seconds=2)


def test_native_autoscaler_cap_reserves_its_control_plane_scale_down_delay() -> None:
    # A native HPA/KEDA target must be capped *before* traffic can make it
    # scale. With a 45-second response reserve, a 120-second budget admits at
    # most three replicas (two billable extras), not five.
    assert native_budget_cap(
        remaining_budget_seconds=120,
        current_ready_replicas=1,
        min_replicas=1,
        max_replicas=5,
        control_interval_seconds=10,
        native_scale_down_reserve_seconds=45,
    ) == 3
    assert native_budget_cap(
        remaining_budget_seconds=100,
        current_ready_replicas=3,
        min_replicas=1,
        max_replicas=5,
        control_interval_seconds=10,
        native_scale_down_reserve_seconds=45,
    ) == 2
