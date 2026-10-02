from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from dap.direct_action_planning_k8s_service_repair.audit import (
    build_inventory,
    verify_inventory,
)
from dap.utils.artifacts import sha256_file, write_json

from .protocol import PROFILES, SEEDS, plan_path, resolve, validate_config


SCHEMA = "dap.k8s.native_comparison_audit.v1"


def _image_digest(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        text=True, capture_output=True, check=True,
    )
    return result.stdout.strip()


def _kubectl(config: dict[str, Any], args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "kubectl", "--context", str(config["kubernetes"]["context"]),
            "-n", str(config["kubernetes"]["namespace"]), *args,
        ],
        text=True, capture_output=True, check=False,
    )


def _prerequisite_evidence(config: dict[str, Any]) -> dict[str, Any]:
    paths = (
        "/apis/metrics.k8s.io/v1beta1",
        "/apis/keda.sh/v1alpha1",
    )
    evidence: dict[str, Any] = {}
    for path in paths:
        result = _kubectl(config, ["get", "--raw", path])
        evidence[path] = {
            "returncode": result.returncode,
            "stdout_sha256": "sha256:" + hashlib.sha256(result.stdout.encode()).hexdigest(),
            "stderr": result.stderr[-1000:],
        }
        if result.returncode != 0:
            raise RuntimeError(f"native autoscaler API unavailable: {path}")
    prometheus = _kubectl(config, ["get", "service", "prometheus", "-o", "name"])
    evidence["prometheus_service"] = {
        "returncode": prometheus.returncode,
        "stdout": prometheus.stdout.strip(),
        "stderr": prometheus.stderr[-1000:],
    }
    if prometheus.returncode != 0:
        raise RuntimeError("Prometheus service unavailable for KEDA")
    return evidence


def _source_paths(project_root: Path) -> list[Path]:
    package = project_root / "src/dap/direct_action_planning_k8s_native_comparison"
    return sorted(package.glob("*.py")) + sorted(
        (project_root / "tests/direct_action_planning_k8s_native_comparison").glob("test_*.py")
    )


def _verify_development_production_inventory(
    project_root: Path, development: Path
) -> dict[str, Any]:
    """Verify frozen production sources while replacing the stale test gate.

    The historical development contract included test files in its source
    inventory.  New terminal-mask tests intentionally change that test tree;
    production code and checkpoints must still match byte-for-byte.
    """

    payload = json.loads(development.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "dap.k8s.service_repair_audit.v1"
        or payload.get("status") != "PASS"
        or payload.get("results_observed") is not False
    ):
        raise ValueError("invalid historical development contract")
    checked: list[dict[str, str]] = []
    for row in payload["source_inventory"]["files"]:
        if not row["path"].startswith("src/"):
            continue
        path = project_root / row["path"]
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise ValueError(f"production source drift: {row['path']}")
        checked.append({"path": row["path"], "sha256": actual})
    if not checked:
        raise ValueError("development contract contains no production sources")
    return {
        "contract_sha256": sha256_file(development),
        "production_sources_checked": checked,
        "historical_test_inventory_replaced": True,
    }


def _input_paths(
    project_root: Path, config: dict[str, Any], config_path: Path
) -> list[Path]:
    paths = [
        config_path,
        project_root / "research/direct_action_planning_k8s_native_comparison/PLAN.md",
        resolve(config_path, config["development_contract"]),
        resolve(config_path, config["paths"]["system_model"]),
        project_root / "src/dap/direct_action_planning_k8s_service_repair/runtime.py",
        project_root / "src/dap/direct_action_planning_k8s_service_repair/planner.py",
        project_root / "research/direct_action_planning_k8s_prototype/baselines/autoscaler_runtime.py",
        project_root / "research/direct_action_planning_k8s_prototype/experiments/run_system_trial.py",
    ]
    for profile in PROFILES:
        values = config["profiles"][profile]
        paths.extend(
            [
                resolve(config_path, values["checkpoint"]),
                resolve(config_path, values["repair_checkpoint"]),
            ]
        )
        for seed in SEEDS:
            plan = plan_path(config, config_path, profile, seed)
            paths.extend([plan, plan.with_suffix(plan.suffix + ".manifest.json")])
    prototype = project_root / "research/direct_action_planning_k8s_prototype/kubernetes"
    paths.extend(sorted(prototype.rglob("*.yaml")))
    return paths


def _run_tests(project_root: Path) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/direct_action_planning_k8s_native_comparison",
        "tests/direct_action_planning_k8s_service_repair",
        "research/direct_action_planning_k8s_prototype/tests/test_action_and_budget.py",
        "research/direct_action_planning_k8s_prototype/tests/test_state_collector.py",
    ]
    result = subprocess.run(
        command, cwd=project_root, text=True, capture_output=True, check=False,
        env={**os.environ, "PYTHONPATH": f"{project_root / 'src'}:{project_root / 'research/direct_action_planning_k8s_prototype'}"},
    )
    evidence = {
        "argv": command,
        "exit_code": result.returncode,
        "stdout": result.stdout[-12000:],
        "stderr": result.stderr[-12000:],
    }
    if result.returncode != 0:
        raise RuntimeError("native-comparison test gate failed")
    return evidence


def freeze_contract(project_root: Path, config_path: Path, output: Path) -> Path:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"append-only contract already exists: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    development = resolve(config_path, config["development_contract"])
    development_evidence = _verify_development_production_inventory(
        project_root, development
    )
    for profile in PROFILES:
        for seed in SEEDS:
            plan = plan_path(config, config_path, profile, seed)
            manifest = json.loads(
                plan.with_suffix(plan.suffix + ".manifest.json").read_text(encoding="utf-8")
            )
            if (
                manifest.get("split") != "test"
                or manifest.get("seed") != seed
                or manifest.get("plan_sha256") != sha256_file(plan)
                or manifest.get("horizon") != 32
                or float(manifest.get("interval_seconds")) != 5.0
            ):
                raise ValueError(f"request-plan mismatch: {plan}")
    contract = {
        "schema": SCHEMA,
        "status": "PASS",
        "results_observed": False,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "authorization_id": "reproducibility-package",
        "config_path": config_path.relative_to(project_root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "source_inventory": build_inventory(_source_paths(project_root), root=project_root),
        "input_inventory": build_inventory(
            _input_paths(project_root, config, config_path), root=project_root
        ),
        "container_image": str(config["kubernetes"]["image"]),
        "container_image_digest": _image_digest(str(config["kubernetes"]["image"])),
        "development_evidence": development_evidence,
        "prerequisites": _prerequisite_evidence(config),
        "test_evidence": _run_tests(project_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, contract)
    return output


def verify_contract(project_root: Path, contract_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != "PASS"
        or contract.get("results_observed") is not False
    ):
        raise ValueError("invalid native-comparison contract")
    config_path = project_root / contract["config_path"]
    if sha256_file(config_path) != contract["config_sha256"]:
        raise ValueError("native-comparison config drift")
    verify_inventory(contract["source_inventory"], root=project_root)
    verify_inventory(contract["input_inventory"], root=project_root)
    if _image_digest(contract["container_image"]) != contract["container_image_digest"]:
        raise ValueError("container image drift")
    if contract.get("test_evidence", {}).get("exit_code") != 0:
        raise ValueError("contract lacks passing tests")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_contract(args.project_root, args.output)
        print("PASS")
    else:
        print(freeze_contract(args.project_root, args.config, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
