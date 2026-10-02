from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dap.direct_action_planning_k8s_cost_calibration.training import (
    aggregate_by_activity,
    stratified_windows,
)


def test_stratified_windows_are_aligned_and_reproducible() -> None:
    kwargs = dict(
        dataset_name="gentd26",
        domain="txt2img",
        split="validation",
        horizon=32,
        activity_quantiles=(0.55, 0.90),
        replicates=2,
        seeds=(101, 102, 103, 104),
        target_peak_rps=80.0,
        max_rps=140.0,
    )
    first = stratified_windows(**kwargs)
    second = stratified_windows(**kwargs)
    assert first[1:] == second[1:]
    assert first[2] == [0.55, 0.55, 0.90, 0.90]
    assert len(first[0]) == len(first[1]) == len(first[2]) == 4
    for left, right in zip(first[0], second[0], strict=True):
        np.testing.assert_array_equal(left, right)


def test_aggregate_by_activity_reports_global_low_and_high() -> None:
    metrics = [
        SimpleNamespace(completion_ratio=1.0, slo_violation_rate=0.0,
                        ready_cost_seconds=2.0, total_reward=1.0,
                        target_changes=1, budget_violation_seconds=0.0),
        SimpleNamespace(completion_ratio=0.8, slo_violation_rate=0.2,
                        ready_cost_seconds=10.0, total_reward=0.0,
                        target_changes=3, budget_violation_seconds=0.0),
        SimpleNamespace(completion_ratio=0.6, slo_violation_rate=0.4,
                        ready_cost_seconds=30.0, total_reward=-1.0,
                        target_changes=5, budget_violation_seconds=0.0),
    ]
    result = aggregate_by_activity(
        metrics, (0.60, 0.80, 0.95), low_max=0.70, high_min=0.90
    )
    assert result["global"]["ready_cost_seconds"] == 14.0
    assert result["low_activity"]["ready_cost_seconds"] == 2.0
    assert result["high_activity"]["ready_cost_seconds"] == 30.0
    assert result["stratum_counts"] == {"global": 3, "low_activity": 1, "high_activity": 1}

