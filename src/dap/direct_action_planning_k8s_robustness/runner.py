"""Append-only executor for the preregistered Kubernetes perturbation matrix."""

from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from typing import Any

from dap.utils.artifacts import sha256_file, write_json

from .connectivity import ProxyBypassedPrepare
from .perturbations import PerturbedCollector
from .protocol import load_protocol, matrix_cells, resolve_path
from .prototype_api import PROJECT_ROOT, PROTOTYPE_ROOT
from .readiness import ReadinessDelayController

import experiments.run_system_trial as system_trial


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _runtime_paths() -> list[Path]:
    package = PROJECT_ROOT / "src/dap/direct_action_planning_k8s_robustness"
    return [
        package / name
        for name in (
            "prototype_api.py", "protocol.py", "perturbations.py", "readiness.py",
            "connectivity.py", "runner.py"
        )
    ]


def _inventory(paths: list[Path], root: Path) -> dict[str, Any]:
    rows = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['sha256']}  {row['path']}\n".encode())
    return {"sha256": "sha256:" + digest.hexdigest(), "files": rows}


def _image_digest(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        text=True, capture_output=True, check=True,
    )
    return result.stdout.strip()


def _plan_path(config: dict[str, Any], config_path: Path, profile: str, seed: int) -> Path:
    return resolve_path(config_path, config["paths"]["plan_root"]) / profile / f"test__seed{seed}.jsonl"


def _source_paths(config: dict[str, Any], config_path: Path) -> list[Path]:
    paths: list[Path] = []
    reference_root = resolve_path(config_path, config["paths"]["reference_run_root"])
    for profile in config["profiles"]:
        checkpoint = resolve_path(config_path, config["profiles"][profile]["checkpoint"])
        paths.append(checkpoint)
        for seed in config["seeds"]:
            plan = _plan_path(config, config_path, profile, int(seed))
            paths.extend([plan, plan.with_suffix(plan.suffix + ".manifest.json")])
            reference = reference_root / f"formal__{profile}__medium__seed{seed}__dap"
            paths.extend([reference / "run_manifest.json", reference / "result.json"])
    paths.append(resolve_path(config_path, config["paths"]["system_model"]))
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"missing frozen robustness source artifacts: {missing[:3]}")
    return paths


def freeze_contract(
    project_root: str | Path, config_path: str | Path, output_path: str | Path
) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"robustness contract is append-only: {output_path}")
    config = load_protocol(config_path)
    contract = {
        "schema": "dap.k8s.robustness_contract.v1",
        "status": "frozen_before_perturbation_execution",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "config": config,
        "runtime_inventory": _inventory(_runtime_paths(), root),
        "prototype_source_tree_sha256": system_trial._tree_sha256(PROTOTYPE_ROOT),
        "source_artifact_inventory": _inventory(_source_paths(config, config_path), root),
        "image_digest": _image_digest(str(config["kubernetes"]["image"])),
        "results_observed": False,
        "historical_reference_results_previously_accessed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, contract)
    return output_path


def _verify_contract(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_perturbation_execution" or contract.get("results_observed"):
        raise ValueError("invalid robustness contract time boundary")
    config = contract["config"]
    config_path = root / contract["config_path"]
    if sha256_file(config_path) != contract["config_sha256"]:
        raise ValueError("robustness config changed after freeze")
    if _inventory(_runtime_paths(), root)["sha256"] != contract["runtime_inventory"]["sha256"]:
        raise ValueError("robustness runtime code changed after freeze")
    if system_trial._tree_sha256(PROTOTYPE_ROOT) != contract["prototype_source_tree_sha256"]:
        raise ValueError("frozen Kubernetes prototype changed")
    if _inventory(_source_paths(config, config_path), root)["sha256"] != contract["source_artifact_inventory"]["sha256"]:
        raise ValueError("source plans, checkpoints, or reference runs changed")
    if _image_digest(str(config["kubernetes"]["image"])) != contract["image_digest"]:
        raise ValueError("container image changed after freeze")
    return contract


def _run_dir(config: dict[str, Any], config_path: Path, profile: str, condition: str, seed: int, attempt: int) -> Path:
    root = resolve_path(config_path, config["paths"]["run_root"])
    name = f"{config['tier']}__{profile}__{condition}__seed{seed}"
    if attempt > 1:
        name += f"__attempt{attempt}"
    return root / name


def _next_attempt(config: dict[str, Any], config_path: Path, profile: str, condition: str, seed: int) -> int:
    attempt = 1
    while _run_dir(config, config_path, profile, condition, seed, attempt).exists():
        attempt += 1
    return attempt


def _completed(config: dict[str, Any], config_path: Path, profile: str, condition: str, seed: int, contract_hash: str) -> bool:
    root = resolve_path(config_path, config["paths"]["run_root"])
    prefix = f"{config['tier']}__{profile}__{condition}__seed{seed}*"
    for manifest_path in sorted(root.glob(prefix + "/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed" and manifest.get("robustness_contract_sha256") == contract_hash:
            return True
    return False


def run_cell(
    project_root: str | Path, contract_path: str | Path, *,
    profile: str, condition: str, seed: int, attempt: int | None = None,
) -> Path:
    root = Path(project_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = _verify_contract(root, contract_path)
    config = contract["config"]
    config_path = root / contract["config_path"]
    if (profile, condition, int(seed)) not in matrix_cells(config):
        raise ValueError("unregistered robustness cell")
    attempt = _next_attempt(config, config_path, profile, condition, seed) if attempt is None else int(attempt)
    run_dir = _run_dir(config, config_path, profile, condition, seed, attempt)
    run_dir.mkdir(parents=True, exist_ok=False)
    events = run_dir / "perturbation_events.jsonl"
    original_factory = system_trial._make_controller
    original_prepare = system_trial._prepare
    collector_holder: dict[str, PerturbedCollector] = {}

    def factory(method: str, controller_config: Any, runtime_config: dict[str, Any]):
        controller = original_factory(method, controller_config, runtime_config)
        if condition in {"observation_lag_1", "metric_dropout_10pct"}:
            wrapper = PerturbedCollector(
                controller.collector, condition=condition,
                horizon=int(config["horizon_steps"]), seed=int(seed), event_path=events,
            )
            controller.collector = wrapper
            collector_holder["collector"] = wrapper
        return controller

    readiness = ReadinessDelayController(
        context=str(config["kubernetes"]["context"]),
        namespace=str(config["kubernetes"]["namespace"]),
        deployment=str(config["kubernetes"]["deployment"]),
    )
    baseline_delay = int(config["conditions"]["readiness_delay_5s"]["baseline_initial_delay_seconds"])
    perturbed_delay = int(config["conditions"]["readiness_delay_5s"]["perturbed_initial_delay_seconds"])
    delivery: dict[str, Any] = {
        "schema": "dap.k8s.perturbation_delivery.v1", "condition": condition,
        "seed": int(seed), "profile": profile,
    }
    connectivity_prepare = ProxyBypassedPrepare(
        original_prepare=original_prepare,
    )
    system_trial._make_controller = factory
    system_trial._prepare = connectivity_prepare
    try:
        current_delay = readiness.current()
        if current_delay != baseline_delay:
            raise RuntimeError(f"readiness baseline is {current_delay}, expected {baseline_delay}")
        if condition == "readiness_delay_5s":
            readiness.set(perturbed_delay)
            delivery["readiness_delay_before_trial_seconds"] = readiness.current()
        plan_path = _plan_path(config, config_path, profile, int(seed))
        metadata = {
            "matrix_row_id": f"robustness:{profile}:{condition}:seed{seed}:dap",
            "run_id": run_dir.name,
            "attempt": int(attempt), "seed_role": "workload_request_plan",
            "seed": int(seed), "perturbation_condition": condition,
            "robustness_contract_sha256": sha256_file(contract_path),
            "robustness_runtime_sha256": contract["runtime_inventory"]["sha256"],
            "historical_reference_only": True,
        }
        stdout = run_dir / "stdout.log"
        stderr = run_dir / "stderr.log"
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            with redirect_stdout(out), redirect_stderr(err):
                asyncio.run(system_trial.run_trial(
                    config=config, config_path=config_path, method="dap", profile=profile,
                    budget=float(config["budget_seconds"]), plan_path=plan_path,
                    run_dir=run_dir, run_metadata=metadata,
                ))
        wrapper = collector_holder.get("collector")
        delivery["applied_events"] = int(wrapper.applied_events) if wrapper else 0
        delivery["registered_events"] = (
            63 if condition == "observation_lag_1"
            else 6 if condition == "metric_dropout_10pct" else None
        )
        if condition in {"observation_lag_1", "metric_dropout_10pct"} and delivery["applied_events"] != delivery["registered_events"]:
            raise RuntimeError(f"perturbation delivery mismatch: {delivery}")
        delivery["status"] = "PASS"
        write_json(run_dir / "perturbation_delivery.json", delivery)
        return run_dir
    except Exception:
        write_json(run_dir / "robustness_failure.json", {
            "status": "failed", "condition": condition, "profile": profile,
            "seed": int(seed), "traceback": traceback.format_exc(),
        })
        raise
    finally:
        system_trial._make_controller = original_factory
        system_trial._prepare = original_prepare
        connectivity_prepare.close()
        try:
            if readiness.current() != baseline_delay:
                readiness.set(baseline_delay)
            write_json(run_dir / "readiness_restoration.json", {
                "status": "PASS", "restored_initial_delay_seconds": readiness.current(),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as error:
            write_json(run_dir / "readiness_restoration_failure.json", {
                "status": "failed", "error": f"{type(error).__name__}: {error}",
            })


def run_matrix(project_root: str | Path, contract_path: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = _verify_contract(root, contract_path)
    config = contract["config"]
    config_path = root / contract["config_path"]
    contract_hash = sha256_file(contract_path)
    completed = skipped = failed = 0
    for ordinal, (profile, condition, seed) in enumerate(matrix_cells(config), start=1):
        print(f"[{ordinal}/18] robustness:{profile}:{condition}:seed{seed}", flush=True)
        if _completed(config, config_path, profile, condition, seed, contract_hash):
            skipped += 1
            continue
        try:
            run_cell(root, contract_path, profile=profile, condition=condition, seed=seed)
            completed += 1
        except Exception as error:
            failed += 1
            print(f"FAILED {profile}/{condition}/{seed}: {type(error).__name__}: {error}", flush=True)
    summary = {
        "schema": "dap.k8s.robustness_execution_summary.v1",
        "completed_this_invocation": completed, "skipped": skipped, "failed": failed,
        "registered_rows": 18, "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    output = resolve_path(config_path, config["paths"]["run_root"]) / "execution_summary.json"
    write_json(output, summary)
    return summary
