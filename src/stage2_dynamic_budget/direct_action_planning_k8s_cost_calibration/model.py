"""Expose the existing DAP expected-cost coefficient without structural drift."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.transition import (
    RuntimeBranchPrediction,
    RuntimeConsistentSystemModel,
)


LEGACY_COST_WEIGHT = 0.05


@dataclass(frozen=True)
class CostWeightedSystemModel:
    """Delegate every branch mechanic to formal-v2 and adjust one coefficient.

    The frozen system model already subtracts
    ``0.05 * expected_cost / total_budget``.  This wrapper replaces only that
    coefficient after the complete legacy branch has been constructed.  State,
    service dynamics, action mechanics, expected cost, feasibility, and the
    next observation therefore remain byte-for-byte identical.
    """

    base_model: RuntimeConsistentSystemModel
    cost_weight: float = LEGACY_COST_WEIGHT

    def __post_init__(self) -> None:
        weight = float(self.cost_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("cost_weight must be finite and non-negative")
        object.__setattr__(self, "cost_weight", weight)

    @classmethod
    def load(
        cls,
        path: str | Path,
        profile: str,
        *,
        slo_seconds: float,
        cost_weight: float,
    ) -> "CostWeightedSystemModel":
        return cls(
            RuntimeConsistentSystemModel.load(
                path, profile, slo_seconds=slo_seconds
            ),
            cost_weight=cost_weight,
        )

    def __getattr__(self, name: str) -> Any:
        # Dataclass fields are resolved before __getattr__; every unmodified
        # model property/method is delegated to the audited formal-v2 model.
        return getattr(self.base_model, name)

    def branch(self, **kwargs: Any) -> RuntimeBranchPrediction:
        branch = self.base_model.branch(**kwargs)
        total_budget = max(float(kwargs["total_budget_seconds"]), 1.0)
        correction = (
            (float(self.cost_weight) - LEGACY_COST_WEIGHT)
            * float(branch.expected_cost_seconds)
            / total_budget
        )
        return replace(branch, reward=float(branch.reward - correction))

