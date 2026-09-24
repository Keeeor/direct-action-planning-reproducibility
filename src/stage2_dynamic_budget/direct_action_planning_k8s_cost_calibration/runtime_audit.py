"""Pre-run contract for A2 plans, runtime code, image, and checkpoint."""

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

from .audit import verify_development_contract, verify_frozen_formal_v2
from .runtime import FROZEN_CHECKPOINT_SHA256, validate_calibration_checkpoint


SCHEMA = "dap.k8s.cost_calibration_runtime_audit.v1"


def resolve(config_path: Path, value: str | Path) -> Path:
    return (config_path.parent / Path(value)).resolve()


def activity_quantile_for(config: dict[str, Any], seed: int) -> float:
    seeds = [int(value) for value in config["seeds"]]
    quantiles = [float(value) for value in config["profile"]["activity_quantiles"]]
    if len(seeds) != len(quantiles):
        raise ValueError("activity quantiles and seeds must align")
    try:
        return quantiles[seeds.index(int(seed))]
    except ValueError as exc:
        raise ValueError(f"unregistered plan seed: {seed}") from exc


def plan_path(config: dict[str, Any], config_path: Path, seed: int) -> Path:
    return resolve(config_path, config["paths"]["plan_root"]) / "gentd_inference" / (
        f"{config['source_split']}__seed{int(seed)}.jsonl"
    )


def validate_runtime_config(config: dict[str, Any]) -> None:
    common = {
        "schema": "dap.k8s.cost_calibration_runtime.v1",
        "horizon_steps": 32,
        "control_interval_seconds": 5,
        "budget_seconds": 256,
        "methods": ["dap_calibrated", "threshold"],
        "development_contract": "../contracts/development_v2_contract.json",
        "actions": {
            "no_op": 1,
            "scale_small": 2,
            "scale_medium": 3,
            "scale_large": 5,
        },
        "baseline_parameters": {
            "threshold": {
                "queue_small": 8,
                "queue_medium": 32,
                "queue_large": 96,
            }
        },
        "readiness": {"baseline_initial_delay_seconds": 1},
    }
    for key, expected in common.items():
        if config.get(key) != expected:
            raise ValueError(f"runtime config {key} drift")
    if config.get("controller_defaults") != {
        "horizon_steps": 32,
        "control_interval_seconds": 5,
        "base_replicas": 1,
        "native_scale_down_reserve_seconds": 45,
    }:
        raise ValueError("A2 controller defaults drift")
    if config.get("workload") != {
        "client_timeout_seconds": 15,
        "connection_limit": 512,
    }:
        raise ValueError("A2 workload client drift")
    if config.get("monitor") != {"interval_seconds": 1}:
        raise ValueError("A2 monitor interval drift")
    expected_profile = {
        "name": "gentd_inference",
        "dataset": "gentd26",
        "domain": "txt2img",
        "checkpoint": "../results/checkpoints_v2/gentd_inference/models.pt",
        "slo_seconds": 1.0,
        "capacity_per_pod_rps": 29.97526124225415,
        "target_peak_rps": 80,
        "max_rps": 140,
        "training_quantile": 0.99,
    }
    profile = dict(config.get("profile", {}))
    quantiles = [float(value) for value in profile.pop("activity_quantiles", [])]
    if profile != expected_profile:
        raise ValueError("runtime profile/checkpoint drift")
    mode = config.get("mode")
    seeds = [int(value) for value in config.get("seeds", [])]
    if len(seeds) != len(set(seeds)):
        raise ValueError("runtime seeds must be unique")
    if mode == "pilot_a2":
        if config.get("source_split") != "validation":
            raise ValueError("A2 pilot must use validation split")
        if seeds != [2026081201, 2026081202] or quantiles != [0.60, 0.95]:
            raise ValueError("A2 pilot plan grid drift")
        if int(config.get("max_attempts", 0)) != 1:
            raise ValueError("A2 pilot max_attempts drift")
        expected_paths = {
            "system_model": "../../direct_action_planning_k8s_prototype/results/calibration/system_model.json",
            "plan_root": "../results/request_plans/pilot_a2",
            "run_root": "../results/runs/pilot_a2",
        }
    elif mode == "formal_a2":
        if config.get("source_split") != "test":
            raise ValueError("A2 formal must use test split")
        if seeds != list(range(2026081211, 2026081231)):
            raise ValueError("A2 formal seed drift")
        expected_quantiles = [
            value for value in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
            for _ in range(2)
        ]
        if quantiles != expected_quantiles:
            raise ValueError("A2 formal activity strata drift")
        if int(config.get("max_attempts", 0)) != 2:
            raise ValueError("A2 formal max_attempts drift")
        expected_paths = {
            "system_model": "../../direct_action_planning_k8s_prototype/results/calibration/system_model.json",
            "plan_root": "../results/request_plans/formal_a2",
            "run_root": "../results/runs/formal_a2",
        }
    else:
        raise ValueError("unregistered A2 runtime mode")
    if config.get("paths") != expected_paths:
        raise ValueError("A2 runtime path drift")
    if config.get("kubernetes") != {
        "context": "kind-rl-lab",
        "namespace": "dap-k8s-prototype",
        "deployment": "dap-worker",
        "image": "dap-k8s-service:local",
    }:
        raise ValueError("A2 Kubernetes target drift")


def _runtime_paths(project_root: Path) -> list[Path]:
    package = project_root / "src/stage2_dynamic_budget/direct_action_planning_k8s_cost_calibration"
    paths = [
        package / name
        for name in (
            "model.py", "audit.py", "runtime.py", "runtime_audit.py", "runner.py"
        )
    ]
    inherited = project_root / "src/stage2_dynamic_budget/direct_action_planning_k8s_service_repair"
    paths.extend(
        inherited / name
        for name in (
            "prototype_api.py", "transition.py", "planner.py", "collector.py",
            "runtime.py", "runtime_audit.py",
        )
    )
    paths.append(PROTOTYPE_ROOT / "experiments/run_system_trial.py")
    paths.extend(
        sorted(
            (project_root / "tests/direct_action_planning_k8s_cost_calibration").glob("test_*.py")
        )
    )
    return paths


def _plan_paths(project_root: Path) -> list[Path]:
    root = project_root / "research/direct_action_planning_k8s_cost_calibration"
    return [
        root / "A2_PROTOCOL.md",
        root / "experiment_matrix_a2.md",
        root / "plan_spec_a2.json",
        root / "failure-tree-a2.json",
        root / "contracts/plan_findings_a2.json",
        root / "contracts/failure_tree_report_a2.json",
    ]


def _input_paths(
    project_root: Path, config: dict[str, Any], config_path: Path
) -> list[Path]:
    paths = [
        resolve(config_path, config["development_contract"]),
        resolve(config_path, config["profile"]["checkpoint"]),
        resolve(config_path, config["paths"]["system_model"]),
    ]
    for seed in config["seeds"]:
        plan = plan_path(config, config_path, int(seed))
        paths.extend([plan, plan.with_suffix(plan.suffix + ".manifest.json")])
    paths.extend(sorted((PROTOTYPE_ROOT / "kubernetes").rglob("*.yaml")))
    return paths


def _run_test_gate(project_root: Path) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/direct_action_planning_k8s_cost_calibration",
        "tests/direct_action_planning_k8s_service_repair",
        "research/direct_action_planning_k8s_prototype/tests/test_action_and_budget.py",
        "research/direct_action_planning_k8s_prototype/tests/test_state_collector.py",
        "tests/test_budget_accounting.py", "tests/test_no_future_leakage.py",
    ]
    result = subprocess.run(
        command, cwd=project_root, text=True, capture_output=True, check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(project_root / "src")},
    )
    evidence = {
        "argv": command, "exit_code": result.returncode,
        "stdout": result.stdout[-12000:], "stderr": result.stderr[-12000:],
    }
    if result.returncode != 0:
        raise RuntimeError("A2 runtime test gate failed:\n" + result.stdout + result.stderr)
    return evidence


def freeze_runtime_contract(
    *, project_root: Path, config_path: Path, output_path: Path
) -> Path:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"A2 runtime contract is append-only: {output_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_runtime_config(config)
    development_path = resolve(config_path, config["development_contract"])
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
    manifests: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        path = plan_path(config, config_path, int(seed))
        manifest = json.loads(
            path.with_suffix(path.suffix + ".manifest.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("dataset") != "gentd26"
            or manifest.get("domain") != "txt2img"
            or manifest.get("split") != config["source_split"]
            or manifest.get("horizon") != 32
            or float(manifest.get("interval_seconds")) != 5.0
            or int(manifest.get("seed")) != int(seed)
            or manifest.get("plan_sha256") != sha256_file(path)
            or manifest.get("window_selection", {}).get("kind") != "activity_quantile"
            or float(manifest["window_selection"].get("activity_quantile"))
            != activity_quantile_for(config, int(seed))
        ):
            raise ValueError(f"A2 request plan manifest mismatch: {path}")
        manifests.append(manifest)
    tests = _run_test_gate(project_root)
    contract = {
        "schema": SCHEMA,
        "status": "PASS",
        "phase": config["mode"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "authorization_id": "reproducibility-package",
        "prior_results_accessed": True,
        "a2_system_results_accessed": False,
        "config_path": config_path.relative_to(project_root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "runtime_inventory": build_inventory(_runtime_paths(project_root), root=project_root),
        "plan_inventory": build_inventory(_plan_paths(project_root), root=project_root),
        "input_inventory": build_inventory(
            _input_paths(project_root, config, config_path), root=project_root
        ),
        "development_contract_sha256": sha256_file(development_path),
        "checkpoint_evidence": checkpoint_evidence,
        "container_image": config["kubernetes"]["image"],
        "container_image_digest": _image_digest(config["kubernetes"]["image"]),
        "plan_count": len(manifests),
        "test_evidence": tests,
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "semantic_assertions": {
            "checkpoint_is_frozen_diagnostic_candidate": True,
            "continuation_weight_is_zero": True,
            "cost_weight_is_1_5": True,
            "hard_mask_and_controller_loop_inherited": "VERIFIED_BY_HASH_AND_TEST",
            "paired_activity_plans": "VERIFIED_BY_MANIFEST",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, contract)
    return output_path


def verify_runtime_contract(
    *, project_root: Path, contract_path: Path,
    expected_config_path: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != "PASS"
        or contract.get("a2_system_results_accessed") is not False
    ):
        raise ValueError("invalid A2 runtime audit contract")
    config_path = project_root / contract["config_path"]
    if expected_config_path is not None and config_path.resolve() != expected_config_path.resolve():
        raise ValueError("A2 runtime contract is bound to another config")
    if sha256_file(config_path) != contract["config_sha256"]:
        raise ValueError("A2 runtime config drift")
    for name in ("runtime_inventory", "plan_inventory", "input_inventory"):
        verify_inventory(contract[name], root=project_root)
    if contract.get("test_evidence", {}).get("exit_code") != 0:
        raise ValueError("A2 runtime contract lacks passing tests")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    development_path = resolve(config_path, config["development_contract"])
    if sha256_file(development_path) != contract["development_contract_sha256"]:
        raise ValueError("A2 development contract drift")
    checkpoint_path = resolve(config_path, config["profile"]["checkpoint"])
    if sha256_file(checkpoint_path) != FROZEN_CHECKPOINT_SHA256:
        raise ValueError("A2 checkpoint drift")
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
        print(freeze_runtime_contract(
            project_root=args.project_root,
            config_path=args.config,
            output_path=args.output,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
