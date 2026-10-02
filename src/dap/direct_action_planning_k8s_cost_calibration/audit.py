"""Fail-closed audit for the independent non-structural calibration branch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import yaml

from dap.direct_action_planning_k8s_service_repair.audit import (
    build_inventory,
    verify_inventory,
)
from dap.direct_action_planning_k8s_service_repair.runtime_audit import (
    verify_runtime_contract,
)
from dap.utils.artifacts import sha256_file, write_json


SCHEMA = "dap.k8s.cost_calibration_audit.v1"
AUTHORIZATION_ID = "reproducibility-package"
EXPECTED_FROZEN_HASHES = {
    "runtime_contract_sha256": "a94bf9587d55545495ceccaca8a9fd6cb5a3cb9f24d18f837b3b77658578232c",
    "checkpoint_sha256": "f4d73998a6ce8d768386d3c3fedfc7f1e30fb01cc206e4c945c14d1e81ae1a2f",
    "trace_sha256": "455f28fd9deb2ea18ee267797f4690fe5e208eaf4efda5d67fb1671b2ca1e233",
    "metadata_sha256": "8d818c0193c3c42ab47269691f3d6578a852c34176ea8ed319b72e16c0292b77",
}


def validate_development_config(config: dict[str, Any]) -> None:
    expected = {
        "schema": "dap.k8s.cost_calibration_development.v1",
        "horizon_steps": 32,
        "control_interval_seconds": 5,
        "budgets_seconds": [128, 256, 416],
        "selection_budget_seconds": 256,
        "candidate_iterations": [2, 4, 7],
        "cost_weights": [0.05, 0.1, 0.2, 0.4, 0.8],
        "continuation_weights": [0.0, 0.1, 0.25, 0.5, 0.75],
        "tie_margins": [0.0, 0.01, 0.03],
        "forecast_strategy": "causal_envelope",
        "forecast_multiplier": 1.1,
        "actions": {
            "no_op": 1,
            "scale_small": 2,
            "scale_medium": 3,
            "scale_large": 5,
        },
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"development config violates frozen {key}")
    profile = config.get("profile", {})
    expected_profile = {
        "name": "gentd_inference",
        "dataset": "gentd26",
        "domain": "txt2img",
        "target_peak_rps": 80,
        "max_rps": 140,
        "slo_seconds": 1.0,
        "primary_comparator": "threshold",
    }
    if profile != expected_profile:
        raise ValueError("development config profile drift")
    quantiles = [float(value) for value in config.get("validation_activity_quantiles", [])]
    if quantiles != [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        raise ValueError("validation activity quantile drift")
    replicates = int(config.get("validation_replicates", 0))
    seeds = [int(value) for value in config.get("validation_seeds", [])]
    if replicates != 2 or len(seeds) != len(quantiles) * replicates:
        raise ValueError("validation seed/replicate mismatch")
    if len(set(seeds)) != len(seeds):
        raise ValueError("validation seeds must be unique")
    paths = config.get("paths", {})
    if paths.get("frozen_formal_v2_contract") != (
        "research/direct_action_planning_k8s_service_repair/contracts/formal_v2_runtime_contract.json"
    ):
        raise ValueError("frozen formal-v2 contract path drift")
    if paths.get("frozen_formal_v2_checkpoint") != (
        "research/direct_action_planning_k8s_service_repair/results/checkpoints_v2/gentd_inference"
    ):
        raise ValueError("frozen formal-v2 checkpoint path drift")
    required_guards = {
        "global_completion_loss_max",
        "global_slo_increase_max",
        "low_activity_quantile_max",
        "low_activity_cost_increase_seconds_max",
        "high_activity_quantile_min",
        "high_activity_completion_loss_max",
        "high_activity_slo_increase_max",
        "global_cost_increase_seconds_floor",
        "global_cost_relative_increase_max",
    }
    if set(config.get("selection_guards", {})) != required_guards:
        raise ValueError("selection guard set drift")


def verify_frozen_formal_v2(project_root: Path) -> dict[str, str]:
    project_root = project_root.resolve()
    paths = {
        "runtime_contract_sha256": project_root
        / "research/direct_action_planning_k8s_service_repair/contracts/formal_v2_runtime_contract.json",
        "checkpoint_sha256": project_root
        / "research/direct_action_planning_k8s_service_repair/results/checkpoints_v2/gentd_inference/models.pt",
        "trace_sha256": project_root / "data/processed/gentd26/txt2img.npz",
        "metadata_sha256": project_root / "data/metadata/gentd26/dataset.json",
    }
    evidence = {
        name: sha256_file(path).removeprefix("sha256:")
        for name, path in paths.items()
    }
    for name, expected in EXPECTED_FROZEN_HASHES.items():
        if evidence[name] != expected:
            raise ValueError(f"frozen formal-v2 input drift: {name}")
    verified = verify_runtime_contract(
        project_root=project_root,
        contract_path=paths["runtime_contract_sha256"],
    )
    if verified.get("status") != "PASS" or verified.get("phase") != "formal":
        raise ValueError("frozen formal-v2 runtime contract is invalid")
    return evidence


def _source_paths(project_root: Path) -> list[Path]:
    package = (
        project_root
        / "src/dap/direct_action_planning_k8s_cost_calibration"
    )
    paths = [
        package / name
        for name in ("__init__.py", "model.py", "selection.py", "training.py", "audit.py")
    ]
    paths.extend(
        sorted(
            (
                project_root
                / "tests/direct_action_planning_k8s_cost_calibration"
            ).glob("test_*.py")
        )
    )
    return paths


def _inherited_core_paths(project_root: Path) -> list[Path]:
    package = (
        project_root
        / "src/dap/direct_action_planning_k8s_service_repair"
    )
    return [
        package / name
        for name in (
            "prototype_api.py",
            "transition.py",
            "planner.py",
            "simulation.py",
            "training.py",
            "runtime_audit.py",
        )
    ] + [
        project_root
        / "research/direct_action_planning_k8s_prototype/controller/action_mapper.py",
        project_root
        / "research/direct_action_planning_k8s_prototype/controller/checkpoint_loader.py",
        project_root
        / "research/direct_action_planning_k8s_prototype/workload/trace_converter.py",
    ]


def _plan_paths(project_root: Path) -> list[Path]:
    root = project_root / "research/direct_action_planning_k8s_cost_calibration"
    return [
        root / "PLAN.md",
        root / "PREREGISTRATION.md",
        root / "experiment_matrix.md",
        root / "plan_spec.json",
        root / "failure-tree.json",
        root / "contracts/plan_findings.json",
        root / "contracts/failure_tree_report.json",
    ]


def _input_paths(project_root: Path) -> list[Path]:
    return [
        project_root
        / "research/direct_action_planning_k8s_prototype/results/calibration/system_model.json",
        project_root / "data/processed/gentd26/txt2img.npz",
        project_root / "data/metadata/gentd26/dataset.json",
        project_root
        / "research/direct_action_planning_k8s_service_repair/contracts/formal_v2_runtime_contract.json",
        project_root
        / "research/direct_action_planning_k8s_service_repair/results/checkpoints_v2/gentd_inference/models.pt",
    ]


def _run_test_gate(project_root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/direct_action_planning_k8s_cost_calibration",
        "tests/direct_action_planning_k8s_service_repair",
        "research/direct_action_planning_k8s_prototype/tests/test_action_and_budget.py",
        "research/direct_action_planning_k8s_prototype/tests/test_state_collector.py",
        "tests/test_budget_accounting.py",
        "tests/test_no_future_leakage.py",
    ]
    result = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(project_root / "src")},
    )
    evidence = {
        "argv": command,
        "exit_code": result.returncode,
        "stdout": result.stdout[-12000:],
        "stderr": result.stderr[-12000:],
    }
    if result.returncode != 0:
        raise RuntimeError("cost-calibration audit test gate failed:\n" + result.stdout + result.stderr)
    return evidence


def freeze_development_contract(
    *, project_root: Path, config_path: Path, output_path: Path
) -> Path:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"audit contract is append-only: {output_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_development_config(config)
    plan_root = project_root / "research/direct_action_planning_k8s_cost_calibration"
    plan_report = json.loads(
        (plan_root / "contracts/plan_findings.json").read_text(encoding="utf-8")
    )
    failure_report = json.loads(
        (plan_root / "contracts/failure_tree_report.json").read_text(encoding="utf-8")
    )
    if plan_report.get("verdict") != "pass" or failure_report.get("status") != "PASS":
        raise RuntimeError("research plan or failure-tree gate did not pass")
    frozen_evidence = verify_frozen_formal_v2(project_root)
    tests = _run_test_gate(project_root)
    contract = {
        "schema": SCHEMA,
        "status": "PASS",
        "phase": "development_training_validation",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "authorization_id": AUTHORIZATION_ID,
        "prior_formal_v2_outcomes_accessed": True,
        "new_locked_replay_outcomes_accessed": False,
        "config_path": config_path.relative_to(project_root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "source_inventory": build_inventory(_source_paths(project_root), root=project_root),
        "inherited_core_inventory": build_inventory(
            _inherited_core_paths(project_root), root=project_root
        ),
        "plan_inventory": build_inventory(_plan_paths(project_root), root=project_root),
        "input_inventory": build_inventory(_input_paths(project_root), root=project_root),
        "frozen_formal_v2_evidence": frozen_evidence,
        "test_evidence": tests,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "semantic_assertions": {
            "core_observation_unchanged": "VERIFIED_BY_INHERITED_HASH",
            "core_action_set_unchanged": "VERIFIED_BY_CONFIG_AND_INHERITED_HASH",
            "branch_mechanics_unchanged": "VERIFIED_BY_GOLD_TEST",
            "hard_budget_mask_unchanged": "VERIFIED_BY_INHERITED_HASH_AND_TEST",
            "value_architecture_unchanged": "VERIFIED_BY_INHERITED_HASH",
            "kubernetes_path_unchanged": "NOT_USED_DURING_DEVELOPMENT",
            "only_existing_cost_coefficient_exposed": "VERIFIED_BY_GOLD_TEST",
            "activity_stratified_selector_frozen": "VERIFIED_BY_TEST",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, contract)
    return output_path


def verify_development_contract(
    *,
    project_root: Path,
    contract_path: Path,
    expected_config_path: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != "PASS"
        or contract.get("new_locked_replay_outcomes_accessed") is not False
    ):
        raise ValueError("invalid cost-calibration audit contract")
    config_path = project_root / contract["config_path"]
    if expected_config_path is not None and config_path.resolve() != expected_config_path.resolve():
        raise ValueError("audit contract is bound to another config")
    if sha256_file(config_path) != contract["config_sha256"]:
        raise ValueError("audited config drift")
    for name in (
        "source_inventory",
        "inherited_core_inventory",
        "plan_inventory",
        "input_inventory",
    ):
        verify_inventory(contract[name], root=project_root)
    if contract.get("test_evidence", {}).get("exit_code") != 0:
        raise ValueError("audit contract lacks passing tests")
    if verify_frozen_formal_v2(project_root) != contract["frozen_formal_v2_evidence"]:
        raise ValueError("frozen formal-v2 evidence drift")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_development_contract(
            project_root=args.project_root,
            contract_path=args.output,
            expected_config_path=args.config,
        )
        print("PASS")
    else:
        print(
            freeze_development_contract(
                project_root=args.project_root,
                config_path=args.config,
                output_path=args.output,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
