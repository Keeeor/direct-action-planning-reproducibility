"""Append-only A2 plan generator and real-system paired runner."""

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

import yaml

from dap.direct_action_planning_k8s_robustness.connectivity import (
    ProxyBypassedPrepare,
)
from dap.direct_action_planning_k8s_robustness.readiness import (
    ReadinessDelayController,
)
from dap.direct_action_planning_k8s_service_repair.prototype_api import (
    PROJECT_ROOT,
    PROTOTYPE_ROOT,
)
from dap.utils.artifacts import sha256_file, write_json

from .runtime import CostCalibratedDAPController
from .runtime_audit import (
    activity_quantile_for,
    plan_path,
    resolve,
    validate_runtime_config,
    verify_runtime_contract,
)


if str(PROTOTYPE_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROTOTYPE_ROOT))
import experiments.run_system_trial as system_trial  # noqa: E402
from workload.trace_converter import build_plan, write_plan  # noqa: E402


def run_directory_matches(run_directory_name: str, run_id: str) -> bool:
    if run_directory_name == run_id:
        return True
    prefix = run_id + "__attempt"
    suffix = run_directory_name.removeprefix(prefix)
    return (
        run_directory_name.startswith(prefix)
        and suffix.isdigit()
        and int(suffix) >= 2
        and str(int(suffix)) == suffix
    )


def generate_plans(config_path: Path) -> list[Path]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_runtime_config(config)
    profile = config["profile"]
    outputs: list[Path] = []
    for seed in config["seeds"]:
        output = plan_path(config, config_path, int(seed))
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
        if output.exists() or manifest_path.exists():
            if not (output.exists() and manifest_path.exists()):
                raise RuntimeError(f"incomplete append-only A2 plan: {output}")
            outputs.append(output)
            continue
        rows, manifest = build_plan(
            dataset_name=profile["dataset"],
            domain=profile["domain"],
            split=config["source_split"],
            horizon=int(config["horizon_steps"]),
            interval_seconds=float(config["control_interval_seconds"]),
            seed=int(seed),
            quantile=float(profile["training_quantile"]),
            target_peak_rps=float(profile["target_peak_rps"]),
            max_rps=float(profile["max_rps"]),
            activity_quantile=activity_quantile_for(config, int(seed)),
        )
        write_plan(rows, manifest, output)
        outputs.append(output)
    return outputs


def matrix_cells(config: dict[str, Any]) -> list[tuple[str, str, int]]:
    return [
        ("gentd_inference", method, int(seed))
        for seed in config["seeds"]
        for method in config["methods"]
    ]


def _run_id(config: dict[str, Any], profile: str, method: str, seed: int) -> str:
    return f"{config['mode']}__{profile}__seed{int(seed)}__{method}"


def _run_dir(
    config: dict[str, Any], config_path: Path, profile: str, method: str,
    seed: int, attempt: int,
) -> Path:
    name = _run_id(config, profile, method, seed)
    if attempt > 1:
        name += f"__attempt{attempt}"
    return resolve(config_path, config["paths"]["run_root"]) / name


def _next_attempt(
    config: dict[str, Any], config_path: Path, profile: str, method: str, seed: int
) -> int:
    attempt = 1
    while _run_dir(config, config_path, profile, method, seed, attempt).exists():
        attempt += 1
    return attempt


def _completed(
    config: dict[str, Any], config_path: Path, profile: str, method: str,
    seed: int, runtime_contract_sha256: str,
) -> bool:
    root = resolve(config_path, config["paths"]["run_root"])
    run_id = _run_id(config, profile, method, seed)
    for path in sorted(root.glob("*/run_manifest.json")):
        if not run_directory_matches(path.parent.name, run_id):
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "completed"
            and manifest.get("runtime_audit_sha256") == runtime_contract_sha256
        ):
            return True
    return False


def audit_calibrated_action_log(run_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "controller/controller_actions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    failures: list[str] = []
    for row in rows:
        feasible_q = {
            action: float(value)
            for action, value in row.get("q_values", {}).items()
            if row.get("feasible", {}).get(action) is True
        }
        selected = row.get("action")
        greedy = row.get("greedy_action")
        if (
            not feasible_q
            or selected not in feasible_q
            or feasible_q[selected] < max(feasible_q.values()) - 1.0e-10
            or greedy != max(feasible_q, key=feasible_q.get)
            or row.get("hard_mask_current_ready_replicas") is None
        ):
            failures.append(f"step {row.get('step')}: action/Q/safety evidence mismatch")
    if len(rows) != 32 or failures:
        raise RuntimeError(
            f"A2 calibrated action-log audit failed: steps={len(rows)}; "
            + "; ".join(failures[:3])
        )
    return {
        "steps": len(rows),
        "q_argmax_verified": len(rows),
        "independent_ready_channels_recorded": len(rows),
    }


def _system_trial_config(config: dict[str, Any]) -> dict[str, Any]:
    runtime = copy.deepcopy(config)
    profile = copy.deepcopy(config["profile"])
    profile.pop("name", None)
    runtime["profiles"] = {"gentd_inference": profile}
    runtime["workload"] = {"gentd_inference": copy.deepcopy(config["workload"])}
    return runtime


def run_cell(
    *, project_root: Path, runtime_contract_path: Path,
    profile: str, method: str, seed: int,
) -> Path:
    project_root = project_root.resolve()
    runtime_contract_path = runtime_contract_path.resolve()
    contract = verify_runtime_contract(
        project_root=project_root, contract_path=runtime_contract_path
    )
    config_path = project_root / contract["config_path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (profile, method, int(seed)) not in matrix_cells(config):
        raise ValueError("unregistered A2 runtime cell")
    attempt = _next_attempt(config, config_path, profile, method, int(seed))
    if attempt > int(config["max_attempts"]):
        raise RuntimeError("A2 maximum append-only attempts exhausted")
    run_dir = _run_dir(config, config_path, profile, method, int(seed), attempt)
    run_dir.mkdir(parents=True, exist_ok=False)
    original_factory = system_trial._make_controller
    original_prepare = system_trial._prepare
    development_contract = resolve(config_path, config["development_contract"])
    trial_config = _system_trial_config(config)

    def factory(
        runtime_method: str, controller_config: Any,
        runtime_config: dict[str, Any],
    ) -> Any:
        if runtime_method == "dap_calibrated":
            return CostCalibratedDAPController(
                controller_config, audit_contract=development_contract
            )
        return original_factory(runtime_method, controller_config, runtime_config)

    readiness = ReadinessDelayController(
        context=config["kubernetes"]["context"],
        namespace=config["kubernetes"]["namespace"],
        deployment=config["kubernetes"]["deployment"],
    )
    baseline_delay = int(config["readiness"]["baseline_initial_delay_seconds"])
    connectivity_prepare = ProxyBypassedPrepare(original_prepare=original_prepare)
    system_trial._make_controller = factory
    system_trial._prepare = connectivity_prepare
    try:
        if readiness.current() != baseline_delay:
            raise RuntimeError("readiness delay is not at A2 baseline")
        metadata = {
            "matrix_row_id": _run_id(config, profile, method, int(seed)),
            "run_id": run_dir.name,
            "attempt": attempt,
            "seed_role": "workload_request_plan",
            "seed": int(seed),
            "activity_quantile": activity_quantile_for(config, int(seed)),
            "a2_evidence_label": "authorized_exploratory_locked_replay",
            "runtime_audit_sha256": sha256_file(runtime_contract_path),
            "development_audit_sha256": sha256_file(development_contract),
            "calibrated_checkpoint_sha256": sha256_file(
                resolve(config_path, config["profile"]["checkpoint"])
            ),
        }
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as out, (
            run_dir / "stderr.log"
        ).open("w", encoding="utf-8") as err:
            with redirect_stdout(out), redirect_stderr(err):
                result = asyncio.run(system_trial.run_trial(
                    config=trial_config,
                    config_path=config_path,
                    method=method,
                    profile=profile,
                    budget=float(config["budget_seconds"]),
                    plan_path=plan_path(config, config_path, int(seed)),
                    run_dir=run_dir,
                    run_metadata=metadata,
                ))
        if float(result["controller"]["budget_violation_seconds"]) > 1.0e-9:
            raise RuntimeError("A2 hard Ready budget violated")
        delivery = {
            "schema": "dap.k8s.cost_calibration_delivery.v1",
            "status": "PASS", "profile": profile, "method": method,
            "seed": int(seed),
            "activity_quantile": activity_quantile_for(config, int(seed)),
            "action_log_audit": (
                audit_calibrated_action_log(run_dir)
                if method == "dap_calibrated" else None
            ),
        }
        write_json(run_dir / "delivery.json", delivery)
        return run_dir
    except Exception:
        write_json(run_dir / "a2_failure.json", {
            "status": "failed", "profile": profile, "method": method,
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
                "status": "PASS",
                "restored_initial_delay_seconds": readiness.current(),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as error:
            write_json(run_dir / "readiness_restoration_failure.json", {
                "status": "failed", "error": f"{type(error).__name__}: {error}",
            })


def run_matrix(project_root: Path, runtime_contract_path: Path) -> dict[str, Any]:
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
    for ordinal, (profile, method, seed) in enumerate(cells, start=1):
        print(f"[{ordinal}/{len(cells)}] {profile}/{method}/seed{seed}", flush=True)
        if _completed(config, config_path, profile, method, seed, runtime_hash):
            skipped += 1
            continue
        try:
            run_cell(
                project_root=project_root,
                runtime_contract_path=runtime_contract_path,
                profile=profile, method=method, seed=seed,
            )
            completed += 1
        except Exception as error:
            failed += 1
            print(
                f"FAILED {profile}/{method}/seed{seed}: "
                f"{type(error).__name__}: {error}", flush=True,
            )
    summary = {
        "schema": "dap.k8s.cost_calibration_execution.v1",
        "mode": config["mode"], "registered_cells": len(cells),
        "completed_this_invocation": completed, "skipped": skipped,
        "failed": failed, "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(
        resolve(config_path, config["paths"]["run_root"]) / "execution_summary.json",
        summary,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--generate-plans", action="store_true")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--method")
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
            project_root=args.project_root,
            runtime_contract_path=args.contract,
            profile=args.profile, method=args.method, seed=args.seed,
        ))
        return 0
    summary = run_matrix(args.project_root, args.contract)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
