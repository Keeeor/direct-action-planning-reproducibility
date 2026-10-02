"""Activity-stratified, cost-aware validation selection frozen before replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SelectionGuards:
    global_completion_loss_max: float = 0.01
    global_slo_increase_max: float = 0.02
    low_activity_cost_increase_seconds_max: float = 20.0
    low_activity_quantile_max: float = 0.70
    high_activity_completion_loss_max: float = 0.01
    high_activity_slo_increase_max: float = 0.02
    high_activity_quantile_min: float = 0.90
    global_cost_increase_seconds_floor: float = 10.0
    global_cost_relative_increase_max: float = 0.05
    budget_violation_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            number = float(value)
            if not np.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.low_activity_quantile_max <= 1.0:
            raise ValueError("low_activity_quantile_max must be in [0, 1]")
        if not 0.0 <= self.high_activity_quantile_min <= 1.0:
            raise ValueError("high_activity_quantile_min must be in [0, 1]")
        if self.low_activity_quantile_max >= self.high_activity_quantile_min:
            raise ValueError("activity strata must be disjoint")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SelectionGuards":
        fields = cls.__dataclass_fields__
        unknown = set(values) - set(fields)
        if unknown:
            raise ValueError(f"unknown selection guard(s): {sorted(unknown)}")
        return cls(**{name: float(value) for name, value in values.items()})


def _finite_metric(group: Mapping[str, Any], name: str) -> float:
    value = float(group[name])
    if not np.isfinite(value):
        raise ValueError(f"nonfinite metric: {name}")
    return value


def _loss(candidate: Mapping[str, Any], comparator: Mapping[str, Any]) -> float:
    return _finite_metric(comparator, "completion_ratio") - _finite_metric(
        candidate, "completion_ratio"
    )


def _slo_increase(
    candidate: Mapping[str, Any], comparator: Mapping[str, Any]
) -> float:
    return _finite_metric(candidate, "slo_violation_rate") - _finite_metric(
        comparator, "slo_violation_rate"
    )


def _cost_increase(
    candidate: Mapping[str, Any], comparator: Mapping[str, Any]
) -> float:
    return _finite_metric(candidate, "ready_cost_seconds") - _finite_metric(
        comparator, "ready_cost_seconds"
    )


def _enrich(
    raw: Mapping[str, Any], comparator: Mapping[str, Any], guards: SelectionGuards
) -> dict[str, Any]:
    row = dict(raw)
    for group in ("global", "low_activity", "high_activity"):
        if group not in raw or group not in comparator:
            raise ValueError(f"candidate and comparator require {group!r} metrics")

    global_metrics = raw["global"]
    low_metrics = raw["low_activity"]
    high_metrics = raw["high_activity"]
    global_comparator = comparator["global"]
    low_comparator = comparator["low_activity"]
    high_comparator = comparator["high_activity"]

    global_completion_loss = _loss(global_metrics, global_comparator)
    global_slo_increase = _slo_increase(global_metrics, global_comparator)
    low_cost_increase = _cost_increase(low_metrics, low_comparator)
    high_completion_loss = _loss(high_metrics, high_comparator)
    high_slo_increase = _slo_increase(high_metrics, high_comparator)
    global_cost_increase = _cost_increase(global_metrics, global_comparator)
    comparator_cost = _finite_metric(global_comparator, "ready_cost_seconds")
    global_cost_allowance = max(
        guards.global_cost_increase_seconds_floor,
        guards.global_cost_relative_increase_max * comparator_cost,
    )
    budget_violation = max(
        _finite_metric(raw[group], "budget_violation_seconds")
        for group in ("global", "low_activity", "high_activity")
    )

    row.update({
        "global_completion_loss": global_completion_loss,
        "global_slo_increase": global_slo_increase,
        "low_activity_cost_increase_seconds": low_cost_increase,
        "high_activity_completion_loss": high_completion_loss,
        "high_activity_slo_increase": high_slo_increase,
        "global_cost_increase_seconds": global_cost_increase,
        "global_cost_allowance_seconds": global_cost_allowance,
        "passed_budget_guard": (
            budget_violation <= guards.budget_violation_tolerance
        ),
        "passed_global_service_guard": (
            global_completion_loss <= guards.global_completion_loss_max + 1.0e-12
            and global_slo_increase <= guards.global_slo_increase_max + 1.0e-12
        ),
        "passed_low_activity_cost_guard": (
            low_cost_increase
            <= guards.low_activity_cost_increase_seconds_max + 1.0e-12
        ),
        "passed_high_activity_service_guard": (
            high_completion_loss
            <= guards.high_activity_completion_loss_max + 1.0e-12
            and high_slo_increase
            <= guards.high_activity_slo_increase_max + 1.0e-12
        ),
        "passed_global_cost_guard": (
            global_cost_increase <= global_cost_allowance + 1.0e-12
        ),
    })
    row["passed_all_guards"] = all(
        bool(row[name])
        for name in (
            "passed_budget_guard",
            "passed_global_service_guard",
            "passed_low_activity_cost_guard",
            "passed_high_activity_service_guard",
            "passed_global_cost_guard",
        )
    )

    # A dimensionless diagnostic violation is used only when no candidate
    # survives. It cannot convert a failed candidate into a passing one.
    row["normalized_guard_violation"] = float(sum((
        max(global_completion_loss - guards.global_completion_loss_max, 0.0)
        / max(guards.global_completion_loss_max, 1.0e-9),
        max(global_slo_increase - guards.global_slo_increase_max, 0.0)
        / max(guards.global_slo_increase_max, 1.0e-9),
        max(low_cost_increase - guards.low_activity_cost_increase_seconds_max, 0.0)
        / max(guards.low_activity_cost_increase_seconds_max, 1.0),
        max(high_completion_loss - guards.high_activity_completion_loss_max, 0.0)
        / max(guards.high_activity_completion_loss_max, 1.0e-9),
        max(high_slo_increase - guards.high_activity_slo_increase_max, 0.0)
        / max(guards.high_activity_slo_increase_max, 1.0e-9),
        max(global_cost_increase - global_cost_allowance, 0.0)
        / max(global_cost_allowance, 1.0),
        max(budget_violation - guards.budget_violation_tolerance, 0.0),
    )))
    return row


def _rank(row: Mapping[str, Any]) -> tuple[float, ...]:
    global_metrics = row["global"]
    service = _finite_metric(global_metrics, "completion_ratio") - _finite_metric(
        global_metrics, "slo_violation_rate"
    )
    return (
        _finite_metric(global_metrics, "ready_cost_seconds"),
        -service,
        _finite_metric(global_metrics, "target_changes"),
        float(row["iteration"]),
        float(row["continuation_weight"]),
        float(row["cost_weight"]),
        float(row["tie_margin"]),
    )


def select_cost_aware_candidate(
    candidates: list[Mapping[str, Any]],
    comparator: Mapping[str, Any],
    guards: SelectionGuards,
) -> dict[str, Any]:
    """Apply the frozen guards and return one fully auditable selection row."""

    if not candidates:
        raise ValueError("candidate list must not be empty")
    evaluated = [_enrich(row, comparator, guards) for row in candidates]
    survivors = [row for row in evaluated if row["passed_all_guards"]]
    if survivors:
        chosen = min(survivors, key=_rank)
        status = "selected_guard_survivor"
    else:
        chosen = min(
            evaluated,
            key=lambda row: (float(row["normalized_guard_violation"]), *_rank(row)),
        )
        status = "diagnostic_no_guard_survivor"
    output = dict(chosen)
    output["selection_status"] = status
    output["selection_diagnostics"] = {
        "candidate_count": len(evaluated),
        "survivor_count": len(survivors),
        "evaluated_candidates": evaluated,
    }
    return output
