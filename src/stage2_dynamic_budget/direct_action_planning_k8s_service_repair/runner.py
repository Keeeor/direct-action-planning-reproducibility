"""Append-only audited runner for pilot, service-cost, and robustness cells."""

from __future__ import annotations

import argparse
import asyncio
import copy
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any

from stage2_dynamic_budget.direct_action_planning_k8s_robustness.connectivity import (
    ProxyBypassedPrepare,
)
from stage2_dynamic_budget.direct_action_planning_k8s_robustness.readiness import (
    ReadinessDelayController,
)
from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json

from .perturbations import PerturbedSemanticCollector
from .prototype_api import PROJECT_ROOT, PROTOTYPE_ROOT
from .runtime import ServiceRepairDAPController
from .runtime_audit import (
    activity_quantile_for,
    plan_path,
    resolve,
    run_directory_matches,
    validate_config,
    verify_runtime_contract,
)


if str(PROTOTYPE_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROTOTYPE_ROOT))
import experiments.run_system_trial as system_trial  # noqa: E402
from workload.trace_converter import build_plan, write_plan  # noqa: E402


PERTURBATIONS = (
    "observation_lag_1", "metric_dropout_10pct", "readiness_delay_5s"
)


def generate_plans(config_path: Path) -> list[Path]:
    """Generate deterministic request plans; no controller or service is run."""

    import yaml

    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    outputs: list[Path] = []
    for profile, settings in config["profiles"].items():
        for seed in config["seeds"]:
            output = plan_path(config, config_path, profile, int(seed))
            manifest_path = output.with_suffix(output.suffix + ".manifest.json")
            if output.exists() or manifest_path.exists():
                if not (output.exists() and manifest_path.exists()):
                    raise RuntimeError(f"incomplete append-only plan: {output}")
                outputs.append(output)
                continue
            rows, manifest = build_plan(
                dataset_name=str(settings["dataset"]),
                domain=str(settings["domain"]),
                split=str(config["source_split"]),
                horizon=int(config["horizon_steps"]),
                interval_seconds=float(config["control_interval_seconds"]),
                seed=int(seed),
                quantile=float(settings.get("training_quantile", 0.99)),
                target_peak_rps=float(settings["target_peak_rps"]),
                max_rps=float(settings["max_rps"]),
                activity_quantile=activity_quantile_for(
                    config, profile, int(seed)
                ),
            )
            write_plan(rows, manifest, output)
            outputs.append(output)
    return outputs


def matrix_cells(config: dict[str, Any]) -> list[tuple[str, str, str, int]]:
    cells: list[tuple[str, str, str, int]] = []
    for profile in config["profiles"]:
        for seed in config["seeds"]:
            for method in config["methods_by_profile"][profile]:
                cells.append((profile, method, "control", int(seed)))
            for condition in config.get("perturbations", []):
                if condition not in PERTURBATIONS:
                    raise ValueError(f"unregistered perturbation: {condition}")
                cells.append((profile, "dap_repaired", condition, int(seed)))
    return cells


def _run_id(
    config: dict[str, Any], profile: str, method: str, condition: str, seed: int
) -> str:
    return f"{config['mode']}__{profile}__{condition}__seed{seed}__{method}"


def _run_dir(
    config: dict[str, Any], config_path: Path, profile: str, method: str,
    condition: str, seed: int, attempt: int,
) -> Path:
    root = resolve(config_path, config["paths"]["run_root"])
    name = _run_id(config, profile, method, condition, seed)
    if attempt > 1:
        name += f"__attempt{attempt}"
    return root / name


def _next_attempt(
    config: dict[str, Any], config_path: Path, profile: str, method: str,
    condition: str, seed: int,
) -> int:
    attempt = 1
    while _run_dir(
        config, config_path, profile, method, condition, seed, attempt
    ).exists():
        attempt += 1
    return attempt


def _completed(
    config: dict[str, Any], config_path: Path, profile: str, method: str,
    condition: str, seed: int, runtime_contract_sha256: str,
) -> bool:
    root = resolve(config_path, config["paths"]["run_root"])
    prefix = _run_id(config, profile, method, condition, seed)
    for path in sorted(root.glob("*/run_manifest.json")):
        if not run_directory_matches(path.parent.name, prefix):
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "completed"
            and manifest.get("runtime_audit_sha256") == runtime_contract_sha256
        ):
            return True
    return False


def _audit_repaired_action_log(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "controller/controller_actions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    failures: list[str] = []
    for row in rows:
        required = (
            "model_ready_replicas", "hard_mask_current_ready_replicas",
            "greedy_action", "q_values", "feasible",
        )
        if any(name not in row for name in required):
            failures.append(f"step {row.get('step')}: missing runtime semantic evidence")
            continue
        feasible_q = {
            action: value for action, value in row["q_values"].items()
            if row["feasible"].get(action)
        }
        if max(feasible_q, key=feasible_q.get) != row["greedy_action"]:
            failures.append(f"step {row['step']}: greedy action is not recorded Q argmax")
    if failures:
        raise RuntimeError("repaired action-log audit failed: " + "; ".join(failures[:3]))
    return {
        "steps": len(rows),
        "q_argmax_verified": len(rows),
        "independent_ready_channels_recorded": len(rows),
    }


def run_cell(
    *, project_root: Path, runtime_contract_path: Path,
    profile: str, method: str, condition: str, seed: int,
) -> Path:
    import yaml

    project_root = project_root.resolve()
    runtime_contract_path = runtime_contract_path.resolve()
    contract = verify_runtime_contract(
        project_root=project_root, contract_path=runtime_contract_path
    )
    config_path = project_root / contract["config_path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (profile, method, condition, int(seed)) not in matrix_cells(config):
        raise ValueError("unregistered runtime cell")
    attempt = _next_attempt(
        config, config_path, profile, method, condition, int(seed)
    )
    run_dir = _run_dir(
        config, config_path, profile, method, condition, int(seed), attempt
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    events = run_dir / "perturbation_events.jsonl"
    original_factory = system_trial._make_controller
    original_prepare = system_trial._prepare
    wrapper_holder: dict[str, PerturbedSemanticCollector] = {}
    development_contract = resolve(config_path, config["development_contract"])

    run_config = copy.deepcopy(config)
    if method == "dap_repaired":
        run_config["profiles"][profile]["checkpoint"] = run_config["profiles"][profile][
            "repair_checkpoint"
        ]

    def factory(
        runtime_method: str, controller_config: Any,
        runtime_config: dict[str, Any],
    ) -> Any:
        if runtime_method != "dap_repaired":
            return original_factory(runtime_method, controller_config, runtime_config)
        controller = ServiceRepairDAPController(
            controller_config, audit_contract=development_contract
        )
        if condition in {"observation_lag_1", "metric_dropout_10pct"}:
            wrapper = PerturbedSemanticCollector(
                controller.collector, condition=condition,
                horizon=int(config["horizon_steps"]), seed=int(seed),
                event_path=events,
            )
            controller.collector = wrapper
            wrapper_holder["collector"] = wrapper
        return controller

    readiness = ReadinessDelayController(
        context=str(config["kubernetes"]["context"]),
        namespace=str(config["kubernetes"]["namespace"]),
        deployment=str(config["kubernetes"]["deployment"]),
    )
    baseline_delay = int(
        config["conditions"]["readiness_delay_5s"]["baseline_initial_delay_seconds"]
    )
    perturbed_delay = int(
        config["conditions"]["readiness_delay_5s"]["perturbed_initial_delay_seconds"]
    )
    connectivity_prepare = ProxyBypassedPrepare(original_prepare=original_prepare)
    system_trial._make_controller = factory
    system_trial._prepare = connectivity_prepare
    delivery: dict[str, Any] = {
        "schema": "dap.k8s.service_repair_delivery.v1",
        "profile": profile, "method": method, "condition": condition,
        "seed": int(seed),
    }
    try:
        if readiness.current() != baseline_delay:
            raise RuntimeError("readiness delay is not at registered baseline")
        if condition == "readiness_delay_5s":
            readiness.set(perturbed_delay)
            delivery["readiness_delay_before_trial_seconds"] = readiness.current()
        metadata = {
            "matrix_row_id": _run_id(config, profile, method, condition, int(seed)),
            "run_id": run_dir.name,
            "attempt": attempt,
            "seed_role": "workload_request_plan",
            "seed": int(seed),
            "perturbation_condition": condition,
            "runtime_audit_sha256": sha256_file(runtime_contract_path),
            "development_audit_sha256": sha256_file(development_contract),
            "repair_checkpoint_sha256": sha256_file(
                resolve(config_path, config["profiles"][profile]["repair_checkpoint"])
            ),
        }
        stdout = run_dir / "stdout.log"
        stderr = run_dir / "stderr.log"
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            with redirect_stdout(out), redirect_stderr(err):
                result = asyncio.run(system_trial.run_trial(
                    config=run_config, config_path=config_path,
                    method=method, profile=profile,
                    budget=float(config["budget_seconds"]),
                    plan_path=plan_path(config, config_path, profile, int(seed)),
                    run_dir=run_dir, run_metadata=metadata,
                ))
        if float(result["controller"]["budget_violation_seconds"]) > 1.0e-9:
            raise RuntimeError("hard Ready budget violated")
        if method == "dap_repaired":
            delivery["action_log_audit"] = _audit_repaired_action_log(run_dir)
        wrapper = wrapper_holder.get("collector")
        delivery["applied_events"] = 0 if wrapper is None else wrapper.applied_events
        expected_events = (
            int(config["horizon_steps"]) - 1
            if condition == "observation_lag_1"
            else int(round(int(config["horizon_steps"]) * 0.10))
            if condition == "metric_dropout_10pct"
            else 0
        )
        if delivery["applied_events"] != expected_events:
            raise RuntimeError(
                f"perturbation delivery mismatch: {delivery['applied_events']} != {expected_events}"
            )
        delivery["status"] = "PASS"
        write_json(run_dir / "delivery.json", delivery)
        return run_dir
    except Exception:
        write_json(run_dir / "service_repair_failure.json", {
            "status": "failed", "profile": profile, "method": method,
            "condition": condition, "seed": int(seed),
            "traceback": traceback.format_exc(),
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
                "status": "PASS",
                "restored_initial_delay_seconds": readiness.current(),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as error:
            write_json(run_dir / "readiness_restoration_failure.json", {
                "status": "failed", "error": f"{type(error).__name__}: {error}",
            })


def run_matrix(project_root: Path, runtime_contract_path: Path) -> dict[str, Any]:
    import yaml

    project_root = project_root.resolve()
    runtime_contract_path = runtime_contract_path.resolve()
    contract = verify_runtime_contract(
        project_root=project_root, contract_path=runtime_contract_path
    )
    config_path = project_root / contract["config_path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime_hash = sha256_file(runtime_contract_path)
    completed = skipped = failed = 0
    cells = matrix_cells(config)
    for ordinal, (profile, method, condition, seed) in enumerate(cells, start=1):
        print(
            f"[{ordinal}/{len(cells)}] {profile}/{method}/{condition}/seed{seed}",
            flush=True,
        )
        if _completed(
            config, config_path, profile, method, condition, seed, runtime_hash
        ):
            skipped += 1
            continue
        try:
            run_cell(
                project_root=project_root,
                runtime_contract_path=runtime_contract_path,
                profile=profile, method=method, condition=condition, seed=seed,
            )
            completed += 1
        except Exception as error:
            failed += 1
            print(
                f"FAILED {profile}/{method}/{condition}/seed{seed}: "
                f"{type(error).__name__}: {error}", flush=True,
            )
    summary = {
        "schema": "dap.k8s.service_repair_execution.v1",
        "mode": config["mode"], "registered_cells": len(cells),
        "completed_this_invocation": completed, "skipped": skipped,
        "failed": failed, "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    output = resolve(config_path, config["paths"]["run_root"]) / "execution_summary.json"
    write_json(output, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--generate-plans", action="store_true")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--method")
    parser.add_argument("--condition", default="control")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.generate_plans:
        if args.config is None:
            raise SystemExit("--config is required with --generate-plans")
        for path in generate_plans(args.config):
            print(path)
        return 0
    if args.contract is None:
        raise SystemExit("--contract is required for execution")
    if args.profile and args.method and args.seed is not None:
        print(run_cell(
            project_root=args.project_root, runtime_contract_path=args.contract,
            profile=args.profile, method=args.method,
            condition=args.condition, seed=args.seed,
        ))
        return 0
    summary = run_matrix(args.project_root, args.contract)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
