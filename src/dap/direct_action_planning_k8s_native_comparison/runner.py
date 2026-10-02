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
from dap.direct_action_planning_k8s_service_repair.planner import (
    RuntimeConsistentPlanner,
)
from dap.direct_action_planning_k8s_service_repair.runtime import (
    ServiceRepairDAPController,
)
import dap.direct_action_planning_k8s_service_repair.runtime as repaired_runtime
from dap.direct_action_planning_k8s_service_repair.runner import (
    _audit_repaired_action_log,
)
from dap.direct_action_planning_k8s_service_repair.prototype_api import (
    PROJECT_ROOT,
    PROTOTYPE_ROOT,
)
from dap.utils.artifacts import sha256_file, write_json

from .audit import _verify_development_production_inventory, verify_contract
from .protocol import matrix_cells, plan_path, resolve, validate_config


if str(PROTOTYPE_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROTOTYPE_ROOT))
import experiments.run_system_trial as system_trial  # noqa: E402


def _run_id(profile: str, method: str, seed: int) -> str:
    return f"native_v1__{profile}__seed{int(seed)}__{method}"


def _run_root(config: dict[str, Any], config_path: Path) -> Path:
    return resolve(config_path, config["paths"]["run_root"])


def _completed(root: Path, run_id: str, contract_hash: str) -> bool:
    for manifest_path in sorted(root.glob(run_id + "*/run_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("status") == "completed"
            and payload.get("native_comparison_contract_sha256") == contract_hash
        ):
            return True
    return False


def _next_run_dir(root: Path, run_id: str) -> Path:
    candidate = root / run_id
    if not candidate.exists():
        return candidate
    attempt = 2
    while (root / f"{run_id}__attempt{attempt}").exists():
        attempt += 1
    return root / f"{run_id}__attempt{attempt}"


def run_cell(
    project_root: Path,
    contract_path: Path,
    profile: str,
    method: str,
    seed: int,
) -> Path:
    project_root = project_root.resolve()
    contract = verify_contract(project_root, contract_path.resolve())
    config_path = project_root / contract["config_path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if (profile, method, int(seed)) not in matrix_cells(config):
        raise ValueError("unregistered native-comparison cell")
    root = _run_root(config, config_path)
    root.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(profile, method, int(seed))
    run_dir = _next_run_dir(root, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = copy.deepcopy(config)
    if method == "dap_repaired":
        run_config["profiles"][profile]["checkpoint"] = run_config["profiles"][profile][
            "repair_checkpoint"
        ]
    original_factory = system_trial._make_controller
    original_prepare = system_trial._prepare
    development = resolve(config_path, config["development_contract"])

    def factory(runtime_method: str, controller_config: Any, runtime_config: dict[str, Any]):
        if runtime_method != "dap_repaired":
            return original_factory(runtime_method, controller_config, runtime_config)
        original_verify = repaired_runtime.verify_contract

        def production_only_verify(*, project_root, contract_path, expected_config_path=None):
            del expected_config_path
            contract_path = Path(contract_path).resolve()
            _verify_development_production_inventory(Path(project_root), contract_path)
            return json.loads(contract_path.read_text(encoding="utf-8"))

        repaired_runtime.verify_contract = production_only_verify
        try:
            return ServiceRepairDAPController(
                controller_config, audit_contract=development
            )
        finally:
            repaired_runtime.verify_contract = original_verify

    prepare = ProxyBypassedPrepare(original_prepare=original_prepare)
    system_trial._make_controller = factory
    system_trial._prepare = prepare
    try:
        metadata = {
            "matrix_row_id": run_id,
            "run_id": run_dir.name,
            "seed_role": "paired_request_plan",
            "seed": int(seed),
            "method_order": [cell[1] for cell in matrix_cells(config) if cell[0] == profile and cell[2] == int(seed)],
            "native_comparison_contract_sha256": sha256_file(contract_path),
            "repair_checkpoint_sha256": sha256_file(
                resolve(config_path, config["profiles"][profile]["repair_checkpoint"])
            ),
        }
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as out, (
            run_dir / "stderr.log"
        ).open("w", encoding="utf-8") as err:
            with redirect_stdout(out), redirect_stderr(err):
                result = asyncio.run(
                    system_trial.run_trial(
                        config=run_config,
                        config_path=config_path,
                        method=method,
                        profile=profile,
                        budget=float(config["budget_seconds"]),
                        plan_path=plan_path(config, config_path, profile, int(seed)),
                        run_dir=run_dir,
                        run_metadata=metadata,
                    )
                )
        violation = float(result["controller"].get("budget_violation_seconds", 0.0))
        if violation > 1.0e-9:
            raise RuntimeError(f"observed Ready-ledger violation: {violation}")
        delivery = {
            "schema": "dap.k8s.native_comparison_delivery.v1",
            "status": "PASS",
            "profile": profile,
            "method": method,
            "seed": int(seed),
            "observed_budget_violation_seconds": violation,
            "completed_requests": int(result["replay"]["completed"]),
        }
        if method == "dap_repaired":
            delivery["action_log_audit"] = _audit_repaired_action_log(run_dir)
        write_json(run_dir / "delivery.json", delivery)
        return run_dir
    except Exception:
        write_json(
            run_dir / "native_comparison_failure.json",
            {
                "status": "failed",
                "profile": profile,
                "method": method,
                "seed": int(seed),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        system_trial._make_controller = original_factory
        system_trial._prepare = original_prepare
        prepare.close()


def run_matrix(project_root: Path, contract_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    contract = verify_contract(project_root, contract_path.resolve())
    config_path = project_root / contract["config_path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _run_root(config, config_path)
    root.mkdir(parents=True, exist_ok=True)
    contract_hash = sha256_file(contract_path)
    completed = skipped = failed = 0
    cells = matrix_cells(config)
    for ordinal, (profile, method, seed) in enumerate(cells, start=1):
        run_id = _run_id(profile, method, seed)
        print(f"[{ordinal}/{len(cells)}] {profile}/{method}/seed{seed}", flush=True)
        if _completed(root, run_id, contract_hash):
            skipped += 1
            continue
        try:
            run_cell(project_root, contract_path, profile, method, seed)
            completed += 1
        except Exception as error:
            failed += 1
            print(f"FAILED {run_id}: {type(error).__name__}: {error}", flush=True)
    summary = {
        "schema": "dap.k8s.native_comparison_execution.v1",
        "registered_cells": len(cells),
        "completed_this_invocation": completed,
        "skipped": skipped,
        "failed": failed,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "execution_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--profile")
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.profile is None and args.method is None and args.seed is None:
        print(json.dumps(run_matrix(args.project_root, args.contract), sort_keys=True))
        return 0
    if None in (args.profile, args.method, args.seed):
        parser.error("profile, method, and seed must be provided together")
    print(run_cell(args.project_root, args.contract, args.profile, args.method, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
