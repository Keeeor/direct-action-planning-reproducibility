"""Second pre-run audit bound to runtime code, plans, image, and checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json

from .audit import build_inventory, verify_contract as verify_development, verify_inventory
from .prototype_api import PROJECT_ROOT, PROTOTYPE_ROOT


SCHEMA = "dap.k8s.service_repair_runtime_audit.v1"


def run_directory_matches(run_directory_name: str, run_id: str) -> bool:
    """Match one exact cell or its numbered retry, never a method prefix."""

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


def audit_run_directory_matching() -> dict[str, bool]:
    """Gold check for the `dap` versus `dap_repaired` prefix collision."""

    run_id = "formal__azure_http__control__seed1__dap"
    checks = {
        "exact": run_directory_matches(run_id, run_id),
        "retry": run_directory_matches(run_id + "__attempt2", run_id),
        "reject_repaired_prefix": not run_directory_matches(
            run_id + "_repaired", run_id
        ),
        "reject_malformed_retry": not run_directory_matches(
            run_id + "__attemptx", run_id
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"run-directory identity audit failed: {checks}")
    return checks


def resolve(config_path: Path, value: str | Path) -> Path:
    return (config_path.parent / Path(value)).resolve()


def plan_path(config: dict[str, Any], config_path: Path, profile: str, seed: int) -> Path:
    return resolve(config_path, config["paths"]["plan_root"]) / profile / (
        f"{config['source_split']}__seed{seed}.jsonl"
    )


def activity_quantile_for(
    config: dict[str, Any], profile: str, seed: int
) -> float | None:
    """Resolve a preregistered load stratum without inspecting run outcomes."""

    settings = config["profiles"][profile]
    if "activity_quantiles" in settings:
        quantiles = [float(value) for value in settings["activity_quantiles"]]
        seeds = [int(value) for value in config["seeds"]]
        if len(quantiles) != len(seeds):
            raise ValueError(f"activity quantile/seed mismatch for {profile}")
        try:
            return quantiles[seeds.index(int(seed))]
        except ValueError as exc:
            raise ValueError(f"unregistered plan seed: {seed}") from exc
    value = settings.get("activity_quantile")
    return None if value is None else float(value)


def _runtime_paths(project_root: Path) -> list[Path]:
    package = project_root / "src/stage2_dynamic_budget/direct_action_planning_k8s_service_repair"
    names = (
        "prototype_api.py", "audit.py", "transition.py", "planner.py",
        "collector.py", "runtime.py", "perturbations.py", "runtime_audit.py",
        "runner.py",
    )
    tests = project_root / "tests/direct_action_planning_k8s_service_repair"
    return [package / name for name in names] + sorted(tests.glob("test_*.py"))


def _input_paths(
    project_root: Path, config: dict[str, Any], config_path: Path
) -> list[Path]:
    paths = [
        resolve(config_path, config["development_contract"]),
        resolve(config_path, config["paths"]["system_model"]),
    ]
    for profile, values in config["profiles"].items():
        paths.extend([
            resolve(config_path, values["checkpoint"]),
            resolve(config_path, values["repair_checkpoint"]),
        ])
        for seed in config["seeds"]:
            plan = plan_path(config, config_path, profile, int(seed))
            paths.extend([plan, plan.with_suffix(plan.suffix + ".manifest.json")])
    paths.extend(sorted((PROTOTYPE_ROOT / "kubernetes").rglob("*.yaml")))
    return paths


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "dap.k8s.service_repair_runtime.v1":
        raise ValueError("unexpected runtime config schema")
    if config.get("source_split") not in {"validation", "test"}:
        raise ValueError("runtime split must be validation or test")
    if config.get("mode") == "pilot" and config.get("source_split") != "validation":
        raise ValueError("pilot must use validation plans")
    if config.get("mode") == "formal" and config.get("source_split") != "test":
        raise ValueError("formal must use test plans")
    if int(config.get("horizon_steps", 0)) != 32:
        raise ValueError("registered runtime horizon is 32")
    if float(config.get("control_interval_seconds", 0)) != 5.0:
        raise ValueError("registered runtime interval is 5 seconds")
    if float(config.get("budget_seconds", 0)) != 256.0:
        raise ValueError("registered runtime budget is 256 Ready seconds")
    if tuple(config.get("profiles", {})) != ("azure_http", "gentd_inference"):
        raise ValueError("both registered profiles are required")
    if len(set(config.get("seeds", []))) != len(config.get("seeds", [])):
        raise ValueError("runtime plan seeds must be unique")
    for profile, settings in config["profiles"].items():
        if "activity_quantile" in settings and "activity_quantiles" in settings:
            raise ValueError(f"ambiguous activity selection for {profile}")
        values = settings.get("activity_quantiles")
        if values is not None:
            if len(values) != len(config.get("seeds", [])):
                raise ValueError(f"activity strata must align with seeds for {profile}")
            if any(not 0.0 <= float(value) <= 1.0 for value in values):
                raise ValueError(f"activity strata outside [0,1] for {profile}")
    if config["mode"] == "pilot":
        if len(config["seeds"]) != 1:
            raise ValueError("pilot requires exactly one validation plan per profile")
        expected_methods = {
            "azure_http": ["dap_repaired", "mpc_4"],
            "gentd_inference": ["dap_repaired", "threshold"],
        }
        if config.get("methods_by_profile") != expected_methods:
            raise ValueError("pilot method set drift")
        if config.get("perturbations") != []:
            raise ValueError("pilot must not include formal perturbation cells")
    elif config["mode"] == "formal":
        if len(config["seeds"]) != 10:
            raise ValueError("formal runtime requires ten locked plans")
        expected = ["dap_repaired", "dap", "static", "threshold", "mpc_4"]
        if any(config["methods_by_profile"].get(name) != expected for name in config["profiles"]):
            raise ValueError("formal method set drift")
        if config.get("perturbations") != [
            "observation_lag_1", "metric_dropout_10pct", "readiness_delay_5s"
        ]:
            raise ValueError("formal perturbation set drift")


def _image_digest(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        text=True, capture_output=True, check=True,
    )
    return result.stdout.strip()


def _run_test_gate(project_root: Path) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/direct_action_planning_k8s_service_repair",
        "research/direct_action_planning_k8s_prototype/tests/test_action_and_budget.py",
        "research/direct_action_planning_k8s_prototype/tests/test_state_collector.py",
        "tests/direct_action_planning_k8s_robustness/test_perturbations.py",
        "tests/direct_action_planning_k8s_robustness/test_connectivity.py",
        "tests/test_budget_accounting.py",
        "tests/test_no_future_leakage.py",
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
        raise RuntimeError("runtime audit test gate failed:\n" + result.stdout + result.stderr)
    return evidence


def freeze_runtime_contract(
    *, project_root: Path, config_path: Path, output_path: Path
) -> Path:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"runtime audit is append-only: {output_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    development_path = resolve(config_path, config["development_contract"])
    verify_development(
        project_root=project_root, contract_path=development_path,
        # The append-only development contract already binds its exact config.
        # Do not silently force later audited repair rounds back to v1.
        expected_config_path=None,
    )
    inputs = _input_paths(project_root, config, config_path)
    tests = _run_test_gate(project_root)
    run_identity = audit_run_directory_matching()
    plan_manifests = []
    for profile in config["profiles"]:
        for seed in config["seeds"]:
            path = plan_path(config, config_path, profile, int(seed))
            manifest = json.loads(
                path.with_suffix(path.suffix + ".manifest.json").read_text(encoding="utf-8")
            )
            if (
                manifest.get("split") != config["source_split"]
                or manifest.get("dataset") != config["profiles"][profile]["dataset"]
                or manifest.get("domain") != config["profiles"][profile]["domain"]
                or manifest.get("horizon") != config["horizon_steps"]
                or float(manifest.get("interval_seconds")) != float(config["control_interval_seconds"])
                or manifest.get("seed") != int(seed)
                or manifest.get("plan_sha256") != sha256_file(path)
            ):
                raise ValueError(f"request plan manifest mismatch: {path}")
            expected_activity = activity_quantile_for(
                config, profile, int(seed)
            )
            if expected_activity is not None and (
                manifest.get("window_selection", {}).get("kind") != "activity_quantile"
                or float(manifest["window_selection"].get("activity_quantile"))
                != float(expected_activity)
            ):
                raise ValueError(f"request plan activity stratum mismatch: {path}")
            plan_manifests.append(manifest)
    contract = {
        "schema": SCHEMA,
        "status": "PASS",
        "phase": config["mode"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "authorization_id": "reproducibility-package",
        "results_observed": False,
        "config_path": config_path.relative_to(project_root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "runtime_inventory": build_inventory(
            _runtime_paths(project_root), root=project_root
        ),
        "input_inventory": build_inventory(inputs, root=project_root),
        "development_contract_sha256": sha256_file(development_path),
        "container_image": str(config["kubernetes"]["image"]),
        "container_image_digest": _image_digest(str(config["kubernetes"]["image"])),
        "plan_count": len(plan_manifests),
        "test_evidence": tests,
        "semantic_assertions": {
            "runtime_uses_separate_model_and_safety_ready": "VERIFIED_BY_TEST",
            "stale_observation_preserves_current_safety_ready": "VERIFIED_BY_TEST",
            "service_utilization_and_ewma_live_adapter": "VERIFIED_BY_TEST",
            "readiness_restoration_required": True,
            "per_seed_activity_strata_match_plan_manifests": "VERIFIED_BY_AUDIT",
            "method_completion_matching_is_exact": run_identity,
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
        or contract.get("results_observed") is not False
    ):
        raise ValueError("invalid runtime audit contract")
    config_path = project_root / contract["config_path"]
    if expected_config_path is not None and config_path.resolve() != expected_config_path.resolve():
        raise ValueError("runtime audit is bound to another config")
    if sha256_file(config_path) != contract["config_sha256"]:
        raise ValueError("runtime config drift")
    verify_inventory(contract["runtime_inventory"], root=project_root)
    verify_inventory(contract["input_inventory"], root=project_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    development_path = resolve(config_path, config["development_contract"])
    if sha256_file(development_path) != contract["development_contract_sha256"]:
        raise ValueError("development contract drift")
    if _image_digest(contract["container_image"]) != contract["container_image_digest"]:
        raise ValueError("container image drift")
    if contract.get("test_evidence", {}).get("exit_code") != 0:
        raise ValueError("runtime audit lacks passing tests")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_runtime_contract(
            project_root=args.project_root, contract_path=args.output,
            expected_config_path=args.config,
        )
        print("PASS")
    else:
        print(freeze_runtime_contract(
            project_root=args.project_root, config_path=args.config,
            output_path=args.output,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
