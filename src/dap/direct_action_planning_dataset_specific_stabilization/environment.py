from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dap.data.trace_windows import select_trace_window
from dap.direct_action_planning_dataset_validation.data import (
    TraceDataset,
)
from dap.envs.synthetic_queue_env import (
    DEFAULT_ACTIONS,
    ActionSpec,
    SyntheticQueueConfig,
)
from dap.envs.trace_driven_env import TraceDrivenQueueEnv


@dataclass(frozen=True)
class DomainActionCalibration:
    source: str
    quantile: float
    quantile_load: float
    base_capacity: float
    capacity_multiplier: float
    action_capacity_deltas: tuple[float, ...]

    def action_specs(self) -> tuple[ActionSpec, ...]:
        return tuple(
            ActionSpec(action.name, action.cost, delta)
            for action, delta in zip(
                DEFAULT_ACTIONS, self.action_capacity_deltas, strict=True
            )
        )


def calibrate_domain_actions(
    training_trace: np.ndarray,
    *,
    quantile: float,
    base_capacity: float,
) -> DomainActionCalibration:
    """Calibrate action throughput from one domain's training trace only."""

    trace = np.asarray(training_trace, dtype=np.float64)
    if trace.ndim != 1 or trace.size == 0:
        raise ValueError("training_trace must be a non-empty one-dimensional array")
    if not np.isfinite(trace).all() or np.any(trace < 0.0):
        raise ValueError("training_trace must contain finite non-negative values")
    q = float(quantile)
    base = float(base_capacity)
    if not np.isfinite(q) or not 0.0 < q <= 1.0:
        raise ValueError("quantile must be finite and in (0, 1]")
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("base_capacity must be finite and positive")
    target = float(np.quantile(trace, q))
    maximum_default_delta = float(DEFAULT_ACTIONS[-1].capacity_delta)
    multiplier = max(1.0, (target - base) / maximum_default_delta)
    deltas = tuple(
        float(action.capacity_delta) * multiplier for action in DEFAULT_ACTIONS
    )
    return DomainActionCalibration(
        source="training_only",
        quantile=q,
        quantile_load=target,
        base_capacity=base,
        capacity_multiplier=multiplier,
        action_capacity_deltas=deltas,
    )


def make_calibrated_trace_env(
    dataset: TraceDataset,
    domain: str,
    split: str,
    *,
    horizon: int,
    budget: float,
    window_seed: int,
    quantile: float,
    base_capacity: float = 5.0,
) -> tuple[TraceDrivenQueueEnv, int, DomainActionCalibration]:
    if domain not in dataset.domains:
        raise ValueError(f"unknown domain: {domain}")
    if "train" not in dataset.domains[domain]:
        raise ValueError(f"domain has no training trace: {domain}")
    trace, start = select_trace_window(
        dataset.domains[domain][split], horizon, window_seed
    )
    calibration = calibrate_domain_actions(
        dataset.domains[domain]["train"],
        quantile=quantile,
        base_capacity=base_capacity,
    )
    config = SyntheticQueueConfig(
        horizon=horizon,
        budget=budget,
        scenario=f"{dataset.name}_{domain}_{split}_capacity_calibrated",
        base_capacity=base_capacity,
        slo_latency=3.0,
        actions=calibration.action_specs(),
    )
    return TraceDrivenQueueEnv(trace, config), start, calibration
