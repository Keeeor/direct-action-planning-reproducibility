"""Append-only audit for scalar-grid amendment A1 (development v2)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import torch
import yaml

from dap.direct_action_planning_k8s_service_repair.audit import (
    build_inventory,
)
from dap.utils.artifacts import sha256_file, write_json

from .audit import (
    SCHEMA,
    _inherited_core_paths,
    _run_test_gate,
    verify_development_contract,
    verify_frozen_formal_v2,
)


AUTHORIZATION_ID = "reproducibility-package"


def validate_development_v2_config(config: dict[str, Any]) -> None:
    expected = {
        "schema": "dap.k8s.cost_calibration_development.v2",
        "seed": 202608101,
        "horizon_steps": 32,
        "control_interval_seconds": 5,
        "budgets_seconds": [128, 256, 416],
        "selection_budget_seconds": 256,
        "gamma": 0.98,
        "train_episodes": 24,
        "validation_replicates": 2,
        "forecaster_epochs": 60,
        "fvi_iterations": 8,
        "candidate_iterations": [2, 4, 7],
        "fvi_epochs_per_iteration": 2,
        "cost_weights": [0.8, 1.0, 1.25, 1.5, 2.0],
        "continuation_weights": [0.0, 0.025, 0.05, 0.075, 0.1],
        "tie_margins": [0.0],
        "forecast_strategy": "causal_envelope",
        "forecast_multiplier": 1.1,
        "learning_rate": 0.001,
        "hidden_dim": 64,
        "actions": {
            "no_op": 1,
            "scale_small": 2,
            "scale_medium": 3,
            "scale_large": 5,
        },
        "baseline_parameters": {"threshold": [8, 32, 96]},
        "selection_guards": {
            "global_completion_loss_max": 0.01,
            "global_slo_increase_max": 0.02,
            "low_activity_quantile_max": 0.70,
            "low_activity_cost_increase_seconds_max": 20.0,
            "high_activity_quantile_min": 0.90,
            "high_activity_completion_loss_max": 0.01,
            "high_activity_slo_increase_max": 0.02,
            "global_cost_increase_seconds_floor": 10.0,
            "global_cost_relative_increase_max": 0.05,
        },
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"development v2 config violates frozen {key}")
    if [float(value) for value in config.get("validation_activity_quantiles", [])] != [
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00
    ]:
        raise ValueError("validation_activity_quantiles drift")
    seeds = [int(value) for value in config.get("validation_seeds", [])]
    if seeds != list(range(2026081101, 2026081121)):
        raise ValueError("validation_seeds drift")
    expected_paths = {
        "system_model": "research/direct_action_planning_k8s_prototype/results/calibration/system_model.json",
        "frozen_formal_v2_contract": "research/direct_action_planning_k8s_service_repair/contracts/formal_v2_runtime_contract.json",
        "frozen_formal_v2_checkpoint": "research/direct_action_planning_k8s_service_repair/results/checkpoints_v2/gentd_inference",
        "checkpoint_root": "research/direct_action_planning_k8s_cost_calibration/results/checkpoints_v2",
    }
    if config.get("paths") != expected_paths:
        raise ValueError("development v2 paths drift")
    expected_profile = {
        "name": "gentd_inference",
        "dataset": "gentd26",
        "domain": "txt2img",
        "target_peak_rps": 80,
        "max_rps": 140,
        "slo_seconds": 1.0,
        "primary_comparator": "threshold",
    }
    if config.get("profile") != expected_profile:
        raise ValueError("development v2 profile drift")


def _source_paths(project_root: Path) -> list[Path]:
    package = (
        project_root
        / "src/dap/direct_action_planning_k8s_cost_calibration"
    )
    paths = [
        package / name
        for name in (
            "__init__.py",
            "model.py",
            "selection.py",
            "training.py",
            "audit.py",
            "audit_v2.py",
        )
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


def _plan_paths(project_root: Path) -> list[Path]:
    root = project_root / "research/direct_action_planning_k8s_cost_calibration"
    return [
        root / "PLAN.md",
        root / "PREREGISTRATION.md",
        root / "AMENDMENT_A1.md",
        root / "experiment_matrix_a1.md",
        root / "plan_spec_a1.json",
        root / "failure-tree-a1.json",
        root / "contracts/plan_findings_a1.json",
        root / "contracts/failure_tree_report_a1.json",
    ]


def _input_paths(project_root: Path) -> list[Path]:
    return [
        project_root
        / "research/direct_action_planning_k8s_cost_calibration/contracts/development_v1_contract.json",
        project_root
        / "research/direct_action_planning_k8s_cost_calibration/results/checkpoints_v1/gentd_inference/models.pt",
        project_root
        / "research/direct_action_planning_k8s_cost_calibration/results/checkpoints_v1/gentd_inference/diagnostics.json",
        project_root
        / "research/direct_action_planning_k8s_prototype/results/calibration/system_model.json",
        project_root / "data/processed/gentd26/txt2img.npz",
        project_root / "data/metadata/gentd26/dataset.json",
        project_root
        / "research/direct_action_planning_k8s_service_repair/contracts/formal_v2_runtime_contract.json",
        project_root
        / "research/direct_action_planning_k8s_service_repair/results/checkpoints_v2/gentd_inference/models.pt",
    ]


def freeze_development_v2_contract(
    *, project_root: Path, config_path: Path, output_path: Path
) -> Path:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"audit contract is append-only: {output_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_development_v2_config(config)
    root = project_root / "research/direct_action_planning_k8s_cost_calibration"
    plan_report = json.loads(
        (root / "contracts/plan_findings_a1.json").read_text(encoding="utf-8")
    )
    failure_report = json.loads(
        (root / "contracts/failure_tree_report_a1.json").read_text(encoding="utf-8")
    )
    if plan_report.get("verdict") != "pass" or failure_report.get("status") != "PASS":
        raise RuntimeError("A1 research plan or failure-tree gate did not pass")
    # Revalidate the complete v1 provenance before using it as an input.
    verify_development_contract(
        project_root=project_root,
        contract_path=root / "contracts/development_v1_contract.json",
        expected_config_path=root / "configs/development_v1.yaml",
    )
    frozen_evidence = verify_frozen_formal_v2(project_root)
    tests = _run_test_gate(project_root)
    contract = {
        "schema": SCHEMA,
        "status": "PASS",
        "phase": "development_training_validation_a1",
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
            "v1_results_preserved": "VERIFIED_BY_INPUT_HASH",
            "selection_margins_unchanged": "VERIFIED_BY_CONFIG_TEST",
            "core_observation_actions_branches_mask_network_unchanged": "VERIFIED_BY_SOURCE_HASH_AND_TEST",
            "only_existing_scalar_grid_refined": "VERIFIED_BY_CONFIG_TEST",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, contract)
    return output_path


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
            freeze_development_v2_contract(
                project_root=args.project_root,
                config_path=args.config,
                output_path=args.output,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

