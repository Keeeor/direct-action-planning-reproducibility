from __future__ import annotations

"""Execute the registered Kubernetes matrix as append-only real-system runs."""

import argparse
import asyncio
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_system_trial import _tree_sha256, run_trial
from workload.trace_converter import build_plan, write_plan


VALID_METHODS = ("static", "threshold", "hpa", "keda", "mpc_4", "dap")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(config_path: Path, value: str | Path) -> Path:
    return (config_path.parent / Path(value)).resolve()


def _frozen_manifest(config: dict[str, Any], config_path: Path) -> Path:
    value = config.get("frozen_config_manifest")
    if not value:
        raise ValueError("formal execution requires a frozen configuration manifest")
    path = _resolve(config_path, str(value))
    if not path.exists():
        raise ValueError(f"frozen configuration manifest is missing: {path}")
    text = path.read_text(encoding="utf-8")
    if "Status: **FROZEN**" not in text:
        raise ValueError("formal execution requires a frozen configuration manifest")
    expected = _sha256(config_path)
    if f"`formal_config_sha256`: `{expected}`" not in text:
        raise ValueError("frozen configuration does not bind the current formal config hash")
    return path


def validate_matrix_config(
    config: dict[str, Any], *, config_path: Path | None = None, require_frozen: bool = False
) -> None:
    if config.get("schema") != "dap.k8s.matrix_config.v1":
        raise ValueError("matrix config must use dap.k8s.matrix_config.v1")
    mode = config.get("mode")
    if mode not in {"pilot", "formal"}:
        raise ValueError("matrix mode must be pilot or formal")
    source_split = config.get("source_split")
    if source_split not in {"validation", "test"}:
        raise ValueError("source_split must be validation or test")
    if mode == "pilot" and source_split != "validation":
        raise ValueError("pilot runs must use validation traces")
    if mode == "formal" and source_split != "test":
        raise ValueError("formal runs must use sealed test traces")
    if require_frozen:
        if mode != "formal":
            raise ValueError("only formal matrices can require a frozen configuration")
        if config_path is None:
            if not config.get("frozen_config_manifest"):
                raise ValueError("formal execution requires a frozen configuration manifest")
        else:
            _frozen_manifest(config, config_path)
    if int(config.get("horizon_steps", 0)) <= 0 or float(config.get("control_interval_seconds", 0)) <= 0:
        raise ValueError("horizon_steps and control_interval_seconds must be positive")
    methods = list(config.get("methods", []))
    if not methods or any(method not in VALID_METHODS for method in methods) or len(set(methods)) != len(methods):
        raise ValueError(f"methods must be an ordered subset of {VALID_METHODS}")
    budgets = config.get("budgets", {})
    if not budgets or any(float(value) <= 0 for value in budgets.values()):
        raise ValueError("budgets must be positive named values")
    seeds = list(config.get("seeds", []))
    if not seeds or len(seeds) != int(config.get("repetitions", 0)) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique and match repetitions")
    profiles = config.get("profiles", {})
    if not profiles:
        raise ValueError("at least one workload profile is required")
    for name, profile in profiles.items():
        required = ("dataset", "domain", "target_peak_rps", "max_rps")
        missing = [field for field in required if field not in profile]
        if missing:
            raise ValueError(f"profile {name!r} lacks {missing}")
        if float(profile["target_peak_rps"]) <= 0 or float(profile["max_rps"]) <= 0:
            raise ValueError(f"profile {name!r} has invalid rate scale")


def _validate_runtime_config(config: dict[str, Any]) -> None:
    required_top = ("kubernetes", "controller_defaults", "paths", "workload", "baseline_parameters")
    missing_top = [field for field in required_top if field not in config]
    if missing_top:
        raise ValueError(f"matrix runtime config lacks {missing_top}")
    if "system_model" not in config["paths"] or "run_root" not in config["paths"] or "plan_root" not in config["paths"]:
        raise ValueError("matrix paths must include system_model, run_root, and plan_root")
    for name, profile in config["profiles"].items():
        required = ("checkpoint", "slo_seconds", "capacity_per_pod_rps")
        missing = [field for field in required if field not in profile]
        if missing:
            raise ValueError(f"runtime profile {name!r} lacks {missing}")


def matrix_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the registered rows while retaining method-paired request seeds."""

    validate_matrix_config(config)
    rows: list[dict[str, Any]] = []
    for profile in config["profiles"]:
        for budget_name, budget_seconds in config["budgets"].items():
            for seed in config["seeds"]:
                for method in config["methods"]:
                    rows.append({
                        "matrix_row_id": f"{config['mode']}:{profile}:{budget_name}:seed{seed}:{method}",
                        "run_id": f"{config['mode']}__{profile}__{budget_name}__seed{seed}__{method}",
                        "mode": config["mode"],
                        "method": method,
                        "profile": profile,
                        "budget_name": budget_name,
                        "budget_seconds": float(budget_seconds),
                        "seed": int(seed),
                    })
    return rows


def _plan_path(config: dict[str, Any], config_path: Path, profile: str, seed: int) -> Path:
    plan_root = _resolve(config_path, config["paths"]["plan_root"])
    return plan_root / profile / f"{config['source_split']}__seed{seed}.jsonl"


def ensure_plan(config: dict[str, Any], config_path: Path, *, profile: str, seed: int) -> Path:
    """Generate one paired request plan, with train-only rate fitting, if absent."""

    path = _plan_path(config, config_path, profile, seed)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    settings = config["profiles"][profile]
    activity_quantiles = settings.get("activity_window_quantiles")
    activity_quantile = None
    if activity_quantiles is not None:
        if len(activity_quantiles) != len(config["seeds"]):
            raise ValueError(f"profile {profile!r} activity_window_quantiles must match repetitions")
        activity_quantile = float(activity_quantiles[list(config["seeds"]).index(seed)])
    expected = {
        "dataset": settings["dataset"],
        "domain": settings["domain"],
        "split": config["source_split"],
        "horizon": int(config["horizon_steps"]),
        "interval_seconds": float(config["control_interval_seconds"]),
        "seed": int(seed),
        "rate_scale": {
            "quantile": float(settings.get("training_quantile", 0.99)),
            "target_peak_rps": float(settings["target_peak_rps"]),
            "max_rps": float(settings["max_rps"]),
        },
        "window_selection": (
            {"kind": "activity_quantile", "activity_quantile": activity_quantile}
            if activity_quantile is not None else {"kind": "seeded_random"}
        ),
    }
    if path.exists() or manifest_path.exists():
        if not (path.exists() and manifest_path.exists()):
            raise RuntimeError(f"incomplete append-only plan bundle: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = {
            "dataset": manifest.get("dataset"), "domain": manifest.get("domain"),
            "split": manifest.get("split"), "horizon": manifest.get("horizon"),
            "interval_seconds": manifest.get("interval_seconds"), "seed": manifest.get("seed"),
            "rate_scale": {key: manifest.get("rate_scale", {}).get(key) for key in expected["rate_scale"]},
            "window_selection": {
                key: manifest.get("window_selection", {}).get(key)
                for key in expected["window_selection"]
            },
        }
        if actual != expected:
            raise RuntimeError(f"existing request plan does not match frozen matrix row: {path}")
        return path
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
        activity_quantile=activity_quantile,
    )
    write_plan(rows, manifest, path)
    return path


def _run_directory(root: Path, run_id: str, attempt: int) -> Path:
    return root / run_id if attempt == 1 else root / f"{run_id}__attempt{attempt}"


def _next_free_attempt(root: Path, run_id: str) -> int:
    """Allocate a new append-only attempt after all historical snapshots."""

    attempt = 1
    while _run_directory(root, run_id, attempt).exists():
        attempt += 1
    return attempt


def _existing_completed(run_root: Path, run_id: str, *, source_tree_sha256: str) -> bool:
    for path in sorted(run_root.glob(f"{run_id}*/run_manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if (
                manifest.get("status") == "completed"
                and manifest.get("source_tree_sha256") == source_tree_sha256
            ):
                return True
        except json.JSONDecodeError:
            continue
    return False


def run_matrix(
    config: dict[str, Any],
    *,
    config_path: Path,
    require_frozen: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    validate_matrix_config(config, config_path=config_path, require_frozen=require_frozen)
    _validate_runtime_config(config)
    if config["mode"] == "formal" and not require_frozen:
        raise ValueError("formal matrix requires --require-frozen")
    if require_frozen:
        _frozen_manifest(config, config_path)
    run_root = _resolve(config_path, config["paths"]["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    source_tree_sha256 = _tree_sha256(ROOT)
    all_rows = matrix_rows(config)
    rows = all_rows
    if limit is not None:
        rows = rows[:max(0, int(limit))]
    matrix_manifest = {
        "schema": "dap.k8s.matrix_execution.v1",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_tree_sha256": source_tree_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": all_rows,
        "registered_rows": len(all_rows),
    }
    matrix_path = run_root / "matrix_manifest.json"
    if matrix_path.exists():
        existing = json.loads(matrix_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != matrix_manifest["config_sha256"]:
            raise RuntimeError("run root is already bound to a different matrix configuration")
        existing_ids = {row.get("run_id") for row in existing.get("rows", [])}
        all_ids = {row.get("run_id") for row in all_rows}
        if not existing_ids.issubset(all_ids):
            raise RuntimeError("existing matrix manifest contains rows outside the current configuration")
        if existing_ids != all_ids:
            matrix_manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            matrix_path.write_text(
                json.dumps(matrix_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        elif existing.get("source_tree_sha256") != source_tree_sha256:
            history = list(existing.get("source_snapshot_history", []))
            if existing.get("source_tree_sha256"):
                history.append({
                    "source_tree_sha256": existing["source_tree_sha256"],
                    "superseded_at": datetime.now(timezone.utc).isoformat(),
                })
            matrix_manifest["source_snapshot_history"] = history
            matrix_manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            matrix_path.write_text(
                json.dumps(matrix_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    else:
        matrix_path.write_text(json.dumps(matrix_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    completed = skipped = failed = 0
    attempts = int(config.get("max_attempts", 2))
    for ordinal, row in enumerate(rows, start=1):
        print(f"[{ordinal}/{len(rows)}] {row['matrix_row_id']}", flush=True)
        if _existing_completed(run_root, row["run_id"], source_tree_sha256=source_tree_sha256):
            skipped += 1
            continue
        plan_path = ensure_plan(config, config_path, profile=row["profile"], seed=row["seed"])
        succeeded = False
        first_attempt = _next_free_attempt(run_root, row["run_id"])
        for attempt in range(first_attempt, first_attempt + attempts):
            run_dir = _run_directory(run_root, row["run_id"], attempt)
            run_dir.mkdir(parents=True, exist_ok=False)
            stdout = run_dir / "stdout.log"
            stderr = run_dir / "stderr.log"
            try:
                with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
                    with redirect_stdout(out), redirect_stderr(err):
                        asyncio.run(run_trial(
                            config=config,
                            config_path=config_path,
                            method=row["method"],
                            profile=row["profile"],
                            budget=row["budget_seconds"],
                            plan_path=plan_path,
                            run_dir=run_dir,
                            run_metadata={
                                "matrix_row_id": row["matrix_row_id"],
                                "run_id": row["run_id"],
                                "attempt": attempt,
                                "seed_role": "workload_request_plan",
                                "seed": row["seed"],
                            },
                        ))
                completed += 1
                succeeded = True
                break
            except Exception as exc:
                with stderr.open("a", encoding="utf-8") as err:
                    err.write(f"matrix runner caught {type(exc).__name__}: {exc}\n")
        if not succeeded:
            failed += 1
    summary = {
        "schema": "dap.k8s.matrix_execution_summary.v1",
        "mode": config["mode"],
        "completed_this_invocation": completed,
        "skipped_completed_this_invocation": skipped,
        "failed_this_invocation": failed,
        "executed_rows_this_invocation": len(rows),
        "registered_rows": len(all_rows),
        "completed_rows_current_source": sum(
            _existing_completed(run_root, row["run_id"], source_tree_sha256=source_tree_sha256)
            for row in all_rows
        ),
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_root / "matrix_execution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary = run_matrix(
        config,
        config_path=config_path,
        require_frozen=args.require_frozen,
        limit=args.limit,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["failed_this_invocation"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
