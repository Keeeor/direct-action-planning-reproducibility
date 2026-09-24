from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectionGuardrails:
    completion_tolerance: float = 0.005
    slo_tolerance: float = 0.01
    cost_budget_fraction: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.completion_tolerance,
            self.slo_tolerance,
            self.cost_budget_fraction,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("selection guardrails must be finite and non-negative")


@dataclass(frozen=True)
class PlanningCandidate:
    candidate_iteration: int
    continuation_weight: float


def select_planning_candidate(
    rows: list[dict],
    *,
    budget: float,
    guardrails: SelectionGuardrails,
) -> tuple[PlanningCandidate, pd.DataFrame]:
    """Select on closed-loop validation after applying immediate-planner guardrails."""

    required = {
        "candidate_iteration",
        "continuation_weight",
        "discounted_return",
        "completion_ratio",
        "slo_violation_rate",
        "total_cost",
    }
    frame = pd.DataFrame(rows)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"selection rows are missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("selection rows are empty")
    numeric = frame[list(required)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("selection rows contain non-finite values")
    summary = (
        frame.groupby(
            ["candidate_iteration", "continuation_weight"], as_index=False
        )[
            [
                "discounted_return",
                "completion_ratio",
                "slo_violation_rate",
                "total_cost",
            ]
        ]
        .mean()
        .sort_values(["candidate_iteration", "continuation_weight"])
        .reset_index(drop=True)
    )
    immediate = summary[np.isclose(summary.continuation_weight, 0.0)]
    if len(immediate) != 1:
        raise ValueError("selection requires exactly one immediate-planner candidate")
    baseline = immediate.iloc[0]
    summary["completion_guardrail"] = (
        summary.completion_ratio
        >= float(baseline.completion_ratio) - guardrails.completion_tolerance
    )
    summary["slo_guardrail"] = (
        summary.slo_violation_rate
        <= float(baseline.slo_violation_rate) + guardrails.slo_tolerance
    )
    summary["cost_guardrail"] = (
        summary.total_cost
        <= float(baseline.total_cost) + guardrails.cost_budget_fraction * float(budget)
    )
    summary["eligible"] = summary[
        ["completion_guardrail", "slo_guardrail", "cost_guardrail"]
    ].all(axis=1)
    eligible = summary[summary.eligible].sort_values(
        [
            "discounted_return",
            "completion_ratio",
            "slo_violation_rate",
            "total_cost",
            "continuation_weight",
            "candidate_iteration",
        ],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    )
    if eligible.empty:
        raise AssertionError("the immediate-planner control must remain eligible")
    best = eligible.iloc[0]
    return (
        PlanningCandidate(
            candidate_iteration=int(best.candidate_iteration),
            continuation_weight=float(best.continuation_weight),
        ),
        summary,
    )

