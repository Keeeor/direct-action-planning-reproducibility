"""Real Kubernetes controller for the frozen A2 calibration checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.collector import (
    RuntimeSemanticCollector,
)
from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.planner import (
    RuntimeConsistentPlanner,
)
from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.prototype_api import (
    BudgetTracker,
    ControllerConfig,
    KubectlClient,
    PROJECT_ROOT,
    StateCollector,
    load_checkpoint,
)
from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.runtime import (
    ServiceRepairDAPController,
)
from stage2_dynamic_budget.utils.artifacts import sha256_file

from .audit import verify_development_contract
from .model import CostWeightedSystemModel


FROZEN_CHECKPOINT_SHA256 = (
    "sha256:8c9ab4d5419856e23d4d71b1bba5efcc7e0cc639f7d7d9bc87b799c65ba62a2c"
)
FROZEN_COST_WEIGHT = 1.5
FROZEN_CONTINUATION_WEIGHT = 0.0
FROZEN_TIE_MARGIN = 0.0


def validate_calibration_checkpoint(
    checkpoint: Any,
    *,
    checkpoint_path: Path,
    development_contract_path: Path,
) -> dict[str, Any]:
    """Fail closed on any A1 checkpoint or method-scope drift."""

    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != FROZEN_CHECKPOINT_SHA256:
        raise ValueError("A2 checkpoint hash drift")
    metadata = dict(getattr(checkpoint, "metadata", {}) or {})
    selection = dict(metadata.get("selection", {}) or {})
    checks = {
        "schema": metadata.get("schema") == "dap.k8s.cost_calibration_selection.v1",
        "profile": metadata.get("profile") == "gentd_inference",
        "core": metadata.get("core_method_changed") is False,
        "cost_weight": np.isclose(
            float(metadata.get("cost_weight", np.nan)), FROZEN_COST_WEIGHT
        ),
        "continuation": np.isclose(
            float(getattr(checkpoint, "continuation_weight", np.nan)),
            FROZEN_CONTINUATION_WEIGHT,
        ),
        "tie_margin": np.isclose(
            float(metadata.get("tie_margin", np.nan)), FROZEN_TIE_MARGIN
        ),
        "iteration": int(selection.get("iteration", -1)) == 2,
        "selection_status": selection.get("selection_status")
        == "diagnostic_no_guard_survivor",
        "forecast": metadata.get("forecast_strategy") == "causal_envelope"
        and np.isclose(float(metadata.get("forecast_multiplier", np.nan)), 1.10),
        "gamma": np.isclose(float(getattr(checkpoint, "gamma", np.nan)), 0.98),
        "development_contract": metadata.get("development_contract_sha256")
        == sha256_file(development_contract_path),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "A2 checkpoint continuation/core/config validation failed: "
            + ", ".join(failed)
        )
    return {
        "checkpoint_sha256": actual_hash,
        "development_contract_sha256": sha256_file(development_contract_path),
        "cost_weight": FROZEN_COST_WEIGHT,
        "continuation_weight": FROZEN_CONTINUATION_WEIGHT,
        "tie_margin": FROZEN_TIE_MARGIN,
        "checks": checks,
    }


def build_calibrated_system_model(
    system_model_path: Path,
    profile: str,
    *,
    slo_seconds: float,
    checkpoint: Any,
) -> CostWeightedSystemModel:
    metadata = dict(getattr(checkpoint, "metadata", {}) or {})
    weight = float(metadata.get("cost_weight", np.nan))
    if not np.isclose(weight, FROZEN_COST_WEIGHT):
        raise ValueError("runtime cost_weight is not the frozen A2 value")
    return CostWeightedSystemModel.load(
        system_model_path,
        profile,
        slo_seconds=slo_seconds,
        cost_weight=weight,
    )


class CostCalibratedDAPController(ServiceRepairDAPController):
    """Inherit the audited loop; replace only checkpoint acceptance/model weight."""

    def __init__(self, config: ControllerConfig, *, audit_contract: Path):
        self.config = config
        self.audit = verify_development_contract(
            project_root=PROJECT_ROOT,
            contract_path=Path(audit_contract),
            expected_config_path=None,
        )
        self.audit_contract_path = Path(audit_contract).resolve()
        self.kube = KubectlClient(
            context=config.context,
            namespace=config.namespace,
            deployment=config.deployment,
        )
        self.checkpoint = load_checkpoint(config.checkpoint_path)
        self.checkpoint_evidence = validate_calibration_checkpoint(
            self.checkpoint,
            checkpoint_path=config.checkpoint_path,
            development_contract_path=self.audit_contract_path,
        )
        self.system_model = build_calibrated_system_model(
            config.system_model_path,
            config.profile,
            slo_seconds=config.slo_seconds,
            checkpoint=self.checkpoint,
        )
        base_collector = StateCollector(
            self.kube,
            capacity_per_pod_rps=config.capacity_per_pod_rps,
            capacity_by_ready_replicas=self.system_model.capacity_by_replicas,
            max_replicas=max(config.action_mapper.targets.values()),
        )
        self.collector: Any = RuntimeSemanticCollector(base_collector)
        self.tracker = BudgetTracker(
            config.total_budget_seconds,
            base_replicas=config.base_replicas,
        )
        self.planner = RuntimeConsistentPlanner(
            checkpoint=self.checkpoint,
            system_model=self.system_model,
            mapper=config.action_mapper,
            control_interval_seconds=config.control_interval_seconds,
            horizon_steps=config.horizon_steps,
            total_budget_seconds=config.total_budget_seconds,
            tie_margin=FROZEN_TIE_MARGIN,
        )
        self.actions_path = config.result_directory / "controller_actions.jsonl"
        self.states_path = config.result_directory / "controller_states.jsonl"
        self.events_path = config.result_directory / "controller_events.jsonl"
