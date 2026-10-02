"""Fail-closed pre-run audit for the A3 locked Kubernetes replay."""

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

from dap.direct_action_planning_k8s_cost_calibration.audit import (
    verify_development_contract,
)
from dap.direct_action_planning_k8s_cost_calibration.runtime import (
    FROZEN_CHECKPOINT_SHA256,
    validate_calibration_checkpoint,
)
from dap.direct_action_planning_k8s_pareto_validation.runtime_audit import (
    verify_runtime_contract as verify_validation_contract,
)
from dap.direct_action_planning_k8s_service_repair.audit import (
    build_inventory,
    verify_inventory,
)
from dap.direct_action_planning_k8s_service_repair.prototype_api import (
    PROJECT_ROOT,
    PROTOTYPE_ROOT,
    load_checkpoint,
)
from dap.direct_action_planning_k8s_service_repair.runtime_audit import (
    _image_digest,
)
from dap.utils.artifacts import sha256_file, write_json

from .protocol import CANDIDATES, activity_quantile_for, validate_runtime_config


SCHEMA = "dap.k8s.pareto_locked_replay_runtime_audit.v1"


def resolve(config_path: Path, value: str | Path) -> Path:
    return (config_path.parent / Path(value)).resolve()


def plan_path(config: dict[str, Any], config_path: Path, seed: int) -> Path:
    return resolve(config_path, config["paths"]["plan_root"]) / "gentd_inference" / (
        f"{config['source_split']}__seed{int(seed)}.jsonl"
    )


def _source_paths(project_root: Path) -> list[Path]:
    package = project_root / "src/dap/direct_action_planning_k8s_pareto_locked_replay"
    paths = sorted(package.glob("*.py"))
    paths.extend(
        sorted(
            (project_root / "tests/direct_action_planning_k8s_pareto_locked_replay").glob("test_*.py")
        )
    )
    inherited = project_root / "src/dap/direct_action_planning_k8s_pareto_calibration"
    paths.extend(inherited / name for name in ("protocol.py", "runtime.py", "runner.py"))
    cost = project_root / "src/dap/direct_action_planning_k8s_cost_calibration"
    paths.extend(cost / name for name in ("model.py", "runtime.py", "runner.py"))
    repair = project_root / "src/dap/direct_action_planning_k8s_service_repair"
    paths.extend(
        repair / name
        for name in ("prototype_api.py", "transition.py", "planner.py", "collector.py", "runtime.py")
    )
    paths.append(PROTOTYPE_ROOT / "experiments/run_system_trial.py")
    return paths


def _plan_paths(project_root: Path, config_path: Path) -> list[Path]:
    branch = project_root / "research/direct_action_planning_k8s_pareto_calibration"
    return [
        branch / "PLAN.md",
        branch / "LOCKED_REPLAY_PROTOCOL.md",
        branch / "LOCKED_REPLAY_DECISIONS.md",
        config_path,
    ]


def _input_paths(
    project_root: Path, config: dict[str, Any], config_path: Path
) -> list[Path]:
    branch = project_root / "research/direct_action_planning_k8s_pareto_calibration"
    paths = [
        resolve(config_path, config["inherited_development_contract"]),
        resolve(config_path, config["profile"]["checkpoint"]),
        resolve(config_path, config["paths"]["system_model"]),
        resolve(config_path, config["validation_selection"]),
        branch / "results/analysis/validation_a3_v1/analysis_audit.json",
        branch / "results/analysis/validation_a3_v1/integrity_report.json",
        branch / "contracts/validation_a3_runtime_contract.json",
    ]
    for seed in config["seeds"]:
        plan = plan_path(config, config_path, int(seed))
        paths.extend([plan, plan.with_suffix(plan.suffix + ".manifest.json")])
    paths.extend(sorted((PROTOTYPE_ROOT / "kubernetes").rglob("*.yaml")))
    return paths


def _verify_validation_selection(
    project_root: Path, config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    branch = project_root / "research/direct_action_planning_k8s_pareto_calibration"
    validation_contract = branch / "contracts/validation_a3_runtime_contract.json"
    verify_validation_contract(
        project_root=project_root,
        contract_path=validation_contract,
        expected_config_path=branch / "configs/validation_a3.yaml",
    )
    decision_path = resolve(config_path, config["validation_selection"])
    decision = json.loads(decision_path.read_text())
    integrity_path = branch / "results/analysis/validation_a3_v1/integrity_report.json"
    audit_path = branch / "results/analysis/validation_a3_v1/analysis_audit.json"
    integrity = json.loads(integrity_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (
        decision.get("inference_allowed") is not False
        or decision.get("selected_candidate") != "dap_cont_0p05"
        or decision.get("locked_replay_authorized_by_frozen_rule") is not True
        or integrity.get("status") != "PASS"
        or audit.get("status") != "PASS"
        or audit.get("validation_decision_sha256") != sha256_file(decision_path)
        or audit.get("integrity_report_sha256") != sha256_file(integrity_path)
    ):
        raise ValueError("A3 validation selection/integrity drift")
    return {
        "validation_decision_sha256": sha256_file(decision_path),
        "validation_integrity_sha256": sha256_file(integrity_path),
        "selected_candidate": "dap_cont_0p05",
    }


def _verify_plan_manifests(config: dict[str, Any], config_path: Path) -> None:
    for seed in config["seeds"]:
        path = plan_path(config, config_path, int(seed))
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if not path.is_file() or not manifest_path.is_file():
            raise ValueError(f"missing A3 locked request plan: {path}")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("dataset") != "gentd26"
            or manifest.get("domain") != "txt2img"
            or manifest.get("split") != "test"
            or int(manifest.get("horizon", -1)) != 32
            or float(manifest.get("interval_seconds", -1)) != 5.0
            or int(manifest.get("seed", -1)) != int(seed)
            or manifest.get("plan_sha256") != sha256_file(path)
            or manifest.get("window_selection", {}).get("kind") != "activity_quantile"
            or not np.isclose(
                float(manifest["window_selection"].get("activity_quantile", -1)),
                activity_quantile_for(config, int(seed)),
            )
        ):
            raise ValueError(f"A3 locked plan manifest mismatch: {path}")


def _run_test_gate(project_root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/direct_action_planning_k8s_pareto_locked_replay",
        "tests/direct_action_planning_k8s_pareto_validation",
        "tests/direct_action_planning_k8s_pareto_calibration",
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
        raise RuntimeError("A3 locked-replay test gate failed:\n" + result.stdout + result.stderr)
    return evidence


def freeze_runtime_contract(
    *, project_root: Path, config_path: Path, output_path: Path
) -> Path:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"locked-replay contract is append-only: {output_path}")
    config = yaml.safe_load(config_path.read_text())
    validate_runtime_config(config)
    selection = _verify_validation_selection(project_root, config, config_path)
    development_path = resolve(config_path, config["inherited_development_contract"])
    verify_development_contract(
        project_root=project_root,
        contract_path=development_path,
        expected_config_path=None,
    )
    checkpoint_path = resolve(config_path, config["profile"]["checkpoint"])
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_evidence = validate_calibration_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        development_contract_path=development_path,
    )
    _verify_plan_manifests(config, config_path)
    tests = _run_test_gate(project_root)
    contract = {
        "schema": SCHEMA,
        "status": "PASS",
        "phase": "locked_replay_a3",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "authorization_id": "reproducibility-package",
        "locked_replay_results_accessed": False,
        "config_path": config_path.relative_to(project_root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "source_inventory": build_inventory(_source_paths(project_root), root=project_root),
        "plan_inventory": build_inventory(_plan_paths(project_root, config_path), root=project_root),
        "input_inventory": build_inventory(_input_paths(project_root, config, config_path), root=project_root),
        "development_contract_sha256": sha256_file(development_path),
        "checkpoint_evidence": checkpoint_evidence,
        "validation_selection": selection,
        "container_image": config["kubernetes"]["image"],
        "container_image_digest": _image_digest(config["kubernetes"]["image"]),
        "plan_count": len(config["seeds"]),
        "test_evidence": tests,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "semantic_assertions": {
            "base_checkpoint_hash_frozen": FROZEN_CHECKPOINT_SHA256,
            "only_selected_candidate": list(CANDIDATES),
            "cost_weight_fixed_1_5": True,
            "continuation_fixed_0_05": True,
            "paired_fresh_test_plans_and_alternating_order": "VERIFIED_BY_CONFIG_AND_MANIFEST",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, contract)
    return output_path


def verify_runtime_contract(
    *, project_root: Path, contract_path: Path, expected_config_path: Path | None = None
) -> dict[str, Any]:
    project_root = project_root.resolve()
    contract = json.loads(Path(contract_path).read_text())
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != "PASS"
        or contract.get("locked_replay_results_accessed") is not False
    ):
        raise ValueError("invalid A3 locked-replay runtime contract")
    config_path = project_root / contract["config_path"]
    if expected_config_path is not None and config_path.resolve() != expected_config_path.resolve():
        raise ValueError("locked-replay contract bound to another config")
    if sha256_file(config_path) != contract["config_sha256"]:
        raise ValueError("locked-replay config drift")
    for name in ("source_inventory", "plan_inventory", "input_inventory"):
        verify_inventory(contract[name], root=project_root)
    if contract.get("test_evidence", {}).get("exit_code") != 0:
        raise ValueError("locked-replay contract lacks passing tests")
    config = yaml.safe_load(config_path.read_text())
    validate_runtime_config(config)
    checkpoint = resolve(config_path, config["profile"]["checkpoint"])
    if sha256_file(checkpoint) != FROZEN_CHECKPOINT_SHA256:
        raise ValueError("locked-replay checkpoint drift")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_runtime_contract(
            project_root=args.project_root,
            contract_path=args.output,
            expected_config_path=args.config,
        )
        print("PASS")
    else:
        print(
            freeze_runtime_contract(
                project_root=args.project_root,
                config_path=args.config,
                output_path=args.output,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
