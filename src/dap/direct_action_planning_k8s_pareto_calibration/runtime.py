"""Continuation-only adapter over the exact audited A2 controller."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from dap.direct_action_planning_k8s_cost_calibration.runtime import (
    CostCalibratedDAPController,
    FROZEN_CHECKPOINT_SHA256,
    FROZEN_COST_WEIGHT,
)
from dap.direct_action_planning_k8s_service_repair.planner import (
    RuntimeConsistentPlanner,
)
from dap.direct_action_planning_k8s_service_repair.prototype_api import (
    ControllerConfig,
    DAPCheckpoint,
)
from dap.utils.artifacts import write_json

from .protocol import ALLOWED_CANDIDATES


ALLOWED_CONTINUATIONS = tuple(ALLOWED_CANDIDATES.values())


def build_candidate_checkpoint(
    checkpoint: DAPCheckpoint, *, continuation_weight: float
) -> DAPCheckpoint:
    """Replace exactly one registered scalar while preserving model identity."""

    weight = float(continuation_weight)
    if not np.isfinite(weight) or not any(
        np.isclose(weight, allowed, rtol=0.0, atol=1.0e-12)
        for allowed in ALLOWED_CONTINUATIONS
    ):
        raise ValueError("continuation_weight is not registered for A3")
    metadata = dict(checkpoint.metadata)
    if (
        checkpoint.sha256 != FROZEN_CHECKPOINT_SHA256
        or metadata.get("schema") != "dap.k8s.cost_calibration_selection.v1"
        or not np.isclose(float(metadata.get("cost_weight", np.nan)), FROZEN_COST_WEIGHT)
        or not np.isclose(float(checkpoint.continuation_weight), 0.0)
    ):
        raise ValueError("A3 base checkpoint/cost/continuation identity drift")
    metadata.update({
        "a3_schema": "dap.k8s.pareto_calibration_candidate.v1",
        "a3_base_checkpoint_sha256": checkpoint.sha256,
        "a3_continuation_weight": weight,
        "a3_core_method_changed": False,
    })
    return replace(
        checkpoint,
        continuation_weight=weight,
        metadata=metadata,
    )


class ParetoCalibratedDAPController(CostCalibratedDAPController):
    """Use the inherited A2 loop and override only its continuation scalar."""

    def __init__(
        self,
        config: ControllerConfig,
        *,
        audit_contract: Path,
        continuation_weight: float,
        candidate_id: str,
    ):
        super().__init__(config, audit_contract=audit_contract)
        expected = ALLOWED_CANDIDATES.get(str(candidate_id))
        if expected is None or not np.isclose(
            float(continuation_weight), expected, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("A3 candidate id/continuation mismatch")
        self.candidate_id = str(candidate_id)
        self.checkpoint = build_candidate_checkpoint(
            self.checkpoint, continuation_weight=float(continuation_weight)
        )
        self.planner = RuntimeConsistentPlanner(
            checkpoint=self.checkpoint,
            system_model=self.system_model,
            mapper=config.action_mapper,
            control_interval_seconds=config.control_interval_seconds,
            horizon_steps=config.horizon_steps,
            total_budget_seconds=config.total_budget_seconds,
            tie_margin=0.0,
        )

    def run(self, *, prepare: bool = True) -> dict[str, Any]:
        result = super().run(prepare=prepare)
        write_json(
            self.config.result_directory / "a3_candidate.json",
            {
                "schema": "dap.k8s.pareto_calibration_candidate.v1",
                "candidate_id": self.candidate_id,
                "base_checkpoint_sha256": self.checkpoint.sha256,
                "cost_weight": FROZEN_COST_WEIGHT,
                "continuation_weight": self.checkpoint.continuation_weight,
                "core_method_changed": False,
            },
        )
        return result

