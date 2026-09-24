"""Fail-closed pre-run audit for A3 plans, code, image, and base checkpoint."""

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

from stage2_dynamic_budget.direct_action_planning_k8s_cost_calibration.audit import (
    verify_development_contract,
    verify_frozen_formal_v2,
)
from stage2_dynamic_budget.direct_action_planning_k8s_cost_calibration.runtime import (
    FROZEN_CHECKPOINT_SHA256,
    validate_calibration_checkpoint,
)
from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.audit import (
    build_inventory,
    verify_inventory,
)
from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.prototype_api import (
    PROJECT_ROOT,
    PROTOTYPE_ROOT,
    load_checkpoint,
)
from stage2_dynamic_budget.direct_action_planning_k8s_service_repair.runtime_audit import (
    _image_digest,
)
from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json

from .protocol import activity_quantile_for, validate_runtime_config


SCHEMA = "dap.k8s.pareto_calibration_runtime_audit.v1"


def resolve(config_path: Path, value: str | Path) -> Path:
    return (config_path.parent / Path(value)).resolve()


def plan_path(config: dict[str, Any], config_path: Path, seed: int) -> Path:
    return resolve(config_path, config["paths"]["plan_root"]) / "gentd_inference" / (
        f"{config['source_split']}__seed{int(seed)}.jsonl"
    )


def _source_paths(project_root: Path) -> list[Path]:
    package = project_root / "src/stage2_dynamic_budget/direct_action_planning_k8s_pareto_calibration"
    paths = sorted(package.glob("*.py"))
    paths.extend(
        sorted(
            (
                project_root
                / "tests/direct_action_planning_k8s_pareto_calibration"
            ).glob("test_*.py")
        )
    )
    inherited = project_root / "src/stage2_dynamic_budget/direct_action_planning_k8s_cost_calibration"
    paths.extend(
        inherited / name
        for name in ("model.py", "runtime.py", "runner.py", "runtime_audit.py")
    )
    repair = project_root / "src/stage2_dynamic_budget/direct_action_planning_k8s_service_repair"
    paths.extend(
        repair / name
        for name in (
            "prototype_api.py", "transition.py", "planner.py", "collector.py",
            "runtime.py",
        )
    )
    paths.append(PROTOTYPE_ROOT / "experiments/run_system_trial.py")
    return paths


def _plan_paths(project_root: Path, config_path: Path) -> list[Path]:
    root = project_root / "research/direct_action_planning_k8s_pareto_calibration"
    return [
        root / "PLAN.md",
        root / "PREREGISTRATION.md",
        root / "DECISIONS.md",
        root / "experiment_matrix.md",
        root / "plan_spec.json",
        root / "failure-tree.json",
        root / "contracts/plan_findings_v3.json",
        root / "contracts/failure_tree_report_v3.json",
        config_path,
    ]


def _input_paths(
    project_root: Path, config: dict[str, Any], config_path: Path
) -> list[Path]:
    paths = [
        resolve(config_path, config["inherited_development_contract"]),
        resolve(config_path, config["profile"]["checkpoint"]),
        resolve(config_path, config["paths"]["system_model"]),
        project_root
        / "research/direct_action_planning_k8s_cost_calibration/contracts/formal_a2_runtime_contract.json",
    ]
    for seed in config["seeds"]:
        plan = plan_path(config, config_path, int(seed))
        paths.extend([plan, plan.with_suffix(plan.suffix + ".manifest.json")])
    paths.extend(sorted((PROTOTYPE_ROOT / "kubernetes").rglob("*.yaml")))
    return paths


def _verify_plan_manifests(config: dict[str, Any], config_path: Path) -> None:
    for seed in config["seeds"]:
        path = plan_path(config, config_path, int(seed))
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if not path.is_file() or not manifest_path.is_file():
            raise ValueError(f"missing A3 request plan: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("dataset") != "gentd26"
            or manifest.get("domain") != "txt2img"
            or manifest.get("split") != "validation"
            or int(manifest.get("horizon", -1)) != 32
            or float(manifest.get("interval_seconds", -1)) != 5.0
            or int(manifest.get("seed", -1)) != int(seed)
            or manifest.get("plan_sha256") != sha256_file(path)
            or manifest.get("window_selection", {}).get("kind")
            != "activity_quantile"
            or not np.isclose(
                float(manifest["window_selection"].get("activity_quantile", -1)),
                activity_quantile_for(config, int(seed)),
            )
        ):
            raise ValueError(f"A3 request plan manifest mismatch: {path}")


def _run_test_gate(project_root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
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
        raise RuntimeError("A3 runtime test gate failed:\n" + result.stdout + result.stderr)
    return evidence


def freeze_runtime_contract(
    *, project_root: Path, config_path: Path, output_path: Path
) -> Path:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"A3 runtime contract is append-only: {output_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_runtime_config(config)
    plan_findings = json.loads(
        (
            project_root
            / "research/direct_action_planning_k8s_pareto_calibration/contracts/plan_findings_v3.json"
        ).read_text(encoding="utf-8")
    )
    failure_report = json.loads(
        (
            project_root
            / "research/direct_action_planning_k8s_pareto_calibration/contracts/failure_tree_report_v3.json"
        ).read_text(encoding="utf-8")
    )
    if plan_findings.get("verdict") != "pass" or failure_report.get("status") != "PASS":
        raise RuntimeError("A3 research plan/failure-tree gate is not passing")
    development_path = resolve(config_path, config["inherited_development_contract"])
    verify_development_contract(
        project_root=project_root,
        contract_path=development_path,
        expected_config_path=None,
    )
    verify_frozen_formal_v2(project_root)
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
        "phase": "screen_a3",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "authorization_id": "reproducibility-package",
        "prior_results_accessed": True,
        "a3_system_results_accessed": False,
        "config_path": config_path.relative_to(project_root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "source_inventory": build_inventory(_source_paths(project_root), root=project_root),
        "plan_inventory": build_inventory(
            _plan_paths(project_root, config_path), root=project_root
        ),
        "input_inventory": build_inventory(
            _input_paths(project_root, config, config_path), root=project_root
        ),
        "development_contract_sha256": sha256_file(development_path),
        "checkpoint_evidence": checkpoint_evidence,
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
            "cost_weight_fixed_1_5": True,
            "only_continuation_scalar_varies": True,
            "hard_mask_branches_controller_inherited": "VERIFIED_BY_HASH_AND_TEST",
            "paired_plans_and_rotated_order": "VERIFIED_BY_CONFIG_AND_MANIFEST",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, contract)
    return output_path


def verify_runtime_contract(
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
        or contract.get("a3_system_results_accessed") is not False
    ):
        raise ValueError("invalid A3 runtime audit contract")
    config_path = project_root / contract["config_path"]
    if expected_config_path is not None and config_path.resolve() != expected_config_path.resolve():
        raise ValueError("A3 runtime contract is bound to another config")
    if sha256_file(config_path) != contract["config_sha256"]:
        raise ValueError("A3 runtime config drift")
    for name in ("source_inventory", "plan_inventory", "input_inventory"):
        verify_inventory(contract[name], root=project_root)
    if contract.get("test_evidence", {}).get("exit_code") != 0:
        raise ValueError("A3 runtime contract lacks passing tests")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_runtime_config(config)
    development = resolve(config_path, config["inherited_development_contract"])
    if sha256_file(development) != contract["development_contract_sha256"]:
        raise ValueError("A3 inherited development contract drift")
    checkpoint = resolve(config_path, config["profile"]["checkpoint"])
    if sha256_file(checkpoint) != FROZEN_CHECKPOINT_SHA256:
        raise ValueError("A3 base checkpoint drift")
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
