"""Frozen three-endpoint A3 screen ranking."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _finite(row: Mapping[str, Any], name: str) -> float:
    value = float(row[name])
    if not np.isfinite(value):
        raise ValueError(f"nonfinite A3 screen metric: {name}")
    return value


def rank_screen_candidates(
    candidates: list[Mapping[str, Any]],
    *,
    completion_loss_max: float,
    slo_increase_max: float,
    ready_cost_increase_seconds_max: float,
    budget_violation_tolerance: float = 1.0e-9,
) -> list[dict[str, Any]]:
    """Rank candidates without converting a failed point guard into success."""

    if not candidates:
        raise ValueError("A3 screen requires at least one candidate")
    margins = (
        float(completion_loss_max),
        float(slo_increase_max),
        float(ready_cost_increase_seconds_max),
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in margins):
        raise ValueError("A3 screen margins must be finite and positive")
    evaluated: list[dict[str, Any]] = []
    for raw in candidates:
        row = dict(raw)
        completion = _finite(row, "completion_loss")
        slo = _finite(row, "slo_increase")
        cost = _finite(row, "ready_cost_increase_seconds")
        budget = _finite(row, "budget_violation_seconds")
        violations = (
            max(completion - margins[0], 0.0) / margins[0],
            max(slo - margins[1], 0.0) / margins[1],
            max(cost - margins[2], 0.0) / margins[2],
            max(budget - budget_violation_tolerance, 0.0)
            / max(budget_violation_tolerance, 1.0e-12),
        )
        row["normalized_guard_violation"] = float(sum(violations))
        row["passed_all_point_guards"] = bool(
            completion <= margins[0] + 1.0e-12
            and slo <= margins[1] + 1.0e-12
            and cost <= margins[2] + 1.0e-12
            and budget <= budget_violation_tolerance
        )
        evaluated.append(row)
    return sorted(
        evaluated,
        key=lambda row: (
            float(row["normalized_guard_violation"]),
            _finite(row, "ready_cost_increase_seconds"),
            _finite(row, "completion_loss") + _finite(row, "slo_increase"),
            _finite(row, "continuation_weight"),
            str(row["method"]),
        ),
    )

