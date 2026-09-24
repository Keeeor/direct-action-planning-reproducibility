"""Fail-closed code audit and content-hash contract for every experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json


SCHEMA = "dap.k8s.service_repair_audit.v1"


def _digest_rows(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['sha256']}  {row['path']}\n".encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def build_inventory(paths: Iterable[Path], *, root: Path) -> dict[str, Any]:
    root = root.resolve()
    rows: list[dict[str, str]] = []
    for raw in sorted({Path(path).resolve() for path in paths}):
        if not raw.is_file():
            raise ValueError(f"missing inventory file: {raw}")
        try:
            relative = raw.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"inventory path is outside root: {raw}") from exc
        rows.append({"path": relative, "sha256": sha256_file(raw)})
    if not rows:
        raise ValueError("inventory must contain at least one file")
    return {"sha256": _digest_rows(rows), "files": rows}


def verify_inventory(inventory: dict[str, Any], *, root: Path) -> None:
    root = root.resolve()
    paths: list[Path] = []
    for row in inventory.get("files", []):
        path = root / row["path"]
        if not path.is_file():
            raise ValueError(f"missing inventory file: {path}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"inventory drift: {row['path']}")
        paths.append(path)
    current = build_inventory(paths, root=root)
    if current["sha256"] != inventory.get("sha256"):
        raise ValueError("inventory drift: aggregate hash")


def _development_source_paths(project_root: Path) -> list[Path]:
    package = project_root / "src/stage2_dynamic_budget/direct_action_planning_k8s_service_repair"
    names = (
        "__init__.py", "prototype_api.py", "transition.py", "planner.py",
        "collector.py", "simulation.py", "training.py", "audit.py",
    )
    tests = project_root / "tests/direct_action_planning_k8s_service_repair"
    return [package / name for name in names] + sorted(tests.glob("test_*.py"))


def _plan_paths(project_root: Path) -> list[Path]:
    root = project_root / "research/direct_action_planning_k8s_service_repair"
    return [
        root / "PLAN.md", root / "PREREGISTRATION.md",
        root / "experiment_matrix.md", root / "failure-tree.json",
        root / "plan_spec.json", root / "contracts/plan_findings_v2.json",
        root / "contracts/plan_gate_v2.json",
        root / "contracts/failure_tree_report_v2.json",
    ]


def _data_paths(project_root: Path) -> list[Path]:
    return [
        project_root / "data/processed/azure_functions_2019/http.npz",
        project_root / "data/processed/azure_functions_2019/preprocessing.json",
        project_root / "data/processed/gentd26/txt2img.npz",
        project_root / "data/metadata/gentd26/dataset.json",
    ]


def _legacy_paths(project_root: Path) -> list[Path]:
    prototype = project_root / "research/direct_action_planning_k8s_prototype"
    return [
        prototype / "results/calibration/system_model.json",
        prototype / "results/prototype_checkpoints_v3/azure_http/models.pt",
        prototype / "results/prototype_checkpoints_v3/gentd_inference/models.pt",
        prototype / "controller/budget_tracker.py",
        prototype / "controller/checkpoint_loader.py",
    ]


def _validate_development_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "dap.k8s.service_repair_development.v1":
        raise ValueError("unexpected development config schema")
    expected = {
        "horizon_steps": 32,
        "control_interval_seconds": 5,
        "selection_budget_seconds": 256,
        "candidate_iterations": [2, 4, 7],
        "continuation_weights": [0.25, 0.5, 0.75, 1.0],
        "tie_margins": [0.0, 0.01, 0.03, 0.05],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"development config violates frozen grid: {key}")
    if tuple(config.get("profiles", {})) != ("azure_http", "gentd_inference"):
        raise ValueError("both registered profiles are required")
    if config["profiles"]["azure_http"]["primary_comparator"] != "mpc_4":
        raise ValueError("Azure primary comparator drift")
    if config["profiles"]["gentd_inference"]["primary_comparator"] != "threshold":
        raise ValueError("GenTD primary comparator drift")
    strategy = str(config.get("forecast_strategy", "learned"))
    multiplier = float(config.get("forecast_multiplier", 1.0))
    if strategy not in {"learned", "causal_envelope"}:
        raise ValueError("unregistered forecast strategy")
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("forecast multiplier must be finite and positive")


def _run_test_gate(project_root: Path) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/direct_action_planning_k8s_service_repair",
        "research/direct_action_planning_k8s_prototype/tests/test_action_and_budget.py",
        "research/direct_action_planning_k8s_prototype/tests/test_state_collector.py",
        "tests/direct_action_planning_k8s_robustness/test_perturbations.py",
        "tests/test_budget_accounting.py",
        "tests/test_no_future_leakage.py",
    ]
    result = subprocess.run(
        command, cwd=project_root, text=True, capture_output=True, check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(project_root / "src")},
    )
    evidence = {
        "argv": command,
        "exit_code": result.returncode,
        "stdout": result.stdout[-12000:],
        "stderr": result.stderr[-12000:],
    }
    if result.returncode != 0:
        raise RuntimeError("pre-run code audit tests failed:\n" + result.stdout + result.stderr)
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
    _validate_development_config(config)
    plan_report = json.loads(
        (project_root / "research/direct_action_planning_k8s_service_repair/contracts/plan_findings_v2.json").read_text(encoding="utf-8")
    )
    failure_report = json.loads(
        (project_root / "research/direct_action_planning_k8s_service_repair/contracts/failure_tree_report_v2.json").read_text(encoding="utf-8")
    )
    if plan_report.get("verdict") != "pass" or failure_report.get("status") != "PASS":
        raise RuntimeError("research plan or failure tree gate did not pass")
    tests = _run_test_gate(project_root)
    contract = {
        "schema": SCHEMA,
        "status": "PASS",
        "phase": "development_training_validation",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "authorization_id": "reproducibility-package",
        "results_observed": False,
        "config_path": config_path.relative_to(project_root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "source_inventory": build_inventory(
            _development_source_paths(project_root), root=project_root
        ),
        "plan_inventory": build_inventory(_plan_paths(project_root), root=project_root),
        "data_inventory": build_inventory(_data_paths(project_root), root=project_root),
        "legacy_input_inventory": build_inventory(
            _legacy_paths(project_root), root=project_root
        ),
        "test_evidence": tests,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "semantic_assertions": {
            "closed_loop_selection_uses_controller_forecast": "VERIFIED_BY_TEST",
            "effective_and_next_ready_are_distinct": "VERIFIED_BY_TEST",
            "one_feasibility_function": "VERIFIED_BY_SOURCE_AND_TEST",
            "current_ready_has_independent_safety_channel": "VERIFIED_BY_TEST",
            "service_utilization_and_ewma_match": "VERIFIED_BY_TEST",
            "causal_forecast_envelope_is_action_invariant": "VERIFIED_BY_TEST",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, contract)
    return output_path


def verify_contract(
    *, project_root: Path, contract_path: Path,
    expected_config_path: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != "PASS"
        or contract.get("results_observed") is not False
    ):
        raise ValueError("invalid pre-run audit contract")
    config_path = project_root / contract["config_path"]
    if expected_config_path is not None and config_path.resolve() != expected_config_path.resolve():
        raise ValueError("audit contract is bound to another config")
    if sha256_file(config_path) != contract["config_sha256"]:
        raise ValueError("audited config drift")
    for name in (
        "source_inventory", "plan_inventory", "data_inventory",
        "legacy_input_inventory",
    ):
        verify_inventory(contract[name], root=project_root)
    if contract.get("test_evidence", {}).get("exit_code") != 0:
        raise ValueError("audit contract lacks passing tests")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_contract(
            project_root=args.project_root, contract_path=args.output,
            expected_config_path=args.config,
        )
        print("PASS")
    else:
        path = freeze_development_contract(
            project_root=args.project_root, config_path=args.config,
            output_path=args.output,
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
