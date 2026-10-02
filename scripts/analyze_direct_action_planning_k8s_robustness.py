from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from dap.direct_action_planning_k8s_robustness.analysis import (
    audit_matrix,
    pair_with_historical,
    read_json,
    select_completed_runs,
)
from dap.direct_action_planning_k8s_robustness.protocol import (
    resolve_path,
)
from dap.direct_action_planning_k8s_robustness.prototype_api import (
    PROTOTYPE_ROOT,  # noqa: F401 - makes the frozen prototype importable
)
from dap.utils.artifacts import sha256_file, write_json

from analysis.aggregate import aggregate_run


PAIR_METRICS = (
    "completion_rate",
    "slo_violation_rate",
    "p95_latency_seconds",
    "p99_latency_seconds",
    "queue_area",
    "ready_replica_seconds",
    "requested_replica_seconds",
    "controller_loop_seconds",
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def exact_bootstrap_ci(values: Iterable[Any]) -> tuple[float, float]:
    array = finite(values)
    if len(array) == 0:
        return math.nan, math.nan
    indices = np.asarray(list(itertools.product(range(len(array)), repeat=len(array))))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def action_latencies(run: Path) -> dict[str, float | int]:
    rows = jsonl(run / "controller" / "controller_actions.jsonl")
    values = finite(row.get("control_loop_latency_seconds") for row in rows)
    return {
        "control_cycles": len(rows),
        "controller_loop_p99_seconds": float(np.quantile(values, 0.99)) if len(values) else math.nan,
        "controller_loop_max_seconds": float(values.max()) if len(values) else math.nan,
    }


def aggregate_perturbations(runs: list[Path], config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        manifest = read_json(run / "run_manifest.json")
        delivery = read_json(run / "perturbation_delivery.json")
        row = aggregate_run(
            run,
            slo_seconds=float(config["profiles"][manifest["profile"]]["slo_seconds"]),
        )
        row.update(
            {
                "condition": manifest["perturbation_condition"],
                "attempt": manifest.get("attempt"),
                "robustness_contract_sha256": manifest["robustness_contract_sha256"],
                "perturbation_applied_events": int(delivery.get("applied_events") or 0),
                "perturbation_registered_events": delivery.get("registered_events"),
                **action_latencies(run),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["profile", "condition", "seed"])


def aggregate_historical(
    config: dict[str, Any], config_path: Path
) -> pd.DataFrame:
    root = resolve_path(config_path, config["paths"]["reference_run_root"])
    rows: list[dict[str, Any]] = []
    for profile in config["profiles"]:
        for seed in config["seeds"]:
            run = root / f"formal__{profile}__medium__seed{seed}__dap"
            row = aggregate_run(
                run, slo_seconds=float(config["profiles"][profile]["slo_seconds"])
            )
            row.update(action_latencies(run))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["profile", "seed"])


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "completion_rate",
        "slo_violation_rate",
        "mean_latency_seconds",
        "p95_latency_seconds",
        "p99_latency_seconds",
        "queue_area",
        "ready_replica_seconds",
        "requested_replica_seconds",
        "unused_budget_seconds",
        "controller_loop_seconds",
        "controller_loop_p99_seconds",
        "controller_loop_max_seconds",
    )
    rows: list[dict[str, Any]] = []
    for (profile, condition), group in frame.groupby(["profile", "condition"], sort=True):
        row: dict[str, Any] = {
            "profile": profile,
            "condition": condition,
            "n_runs": len(group),
        }
        for metric in metrics:
            values = finite(group[metric])
            low, high = exact_bootstrap_ci(values)
            row.update(
                {
                    f"{metric}_mean": float(values.mean()) if len(values) else math.nan,
                    f"{metric}_std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    f"{metric}_ci95_low": low,
                    f"{metric}_ci95_high": high,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    delta_columns = [name for name in frame if name.endswith("_delta")]
    rows: list[dict[str, Any]] = []
    for (profile, condition), group in frame.groupby(["profile", "condition"], sort=True):
        row: dict[str, Any] = {
            "profile": profile,
            "condition": condition,
            "n_paired_runs": len(group),
            "comparison_design": "historical_matched_plan_descriptive_only",
        }
        for column in delta_columns:
            values = finite(group[column])
            low, high = exact_bootstrap_ci(values)
            row.update(
                {
                    f"{column}_mean": float(values.mean()) if len(values) else math.nan,
                    f"{column}_ci95_low": low,
                    f"{column}_ci95_high": high,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def all_action_latency(runs: list[Path]) -> dict[str, Any]:
    values: list[float] = []
    cycles = 0
    for run in runs:
        rows = jsonl(run / "controller" / "controller_actions.jsonl")
        cycles += len(rows)
        values.extend(
            float(row["control_loop_latency_seconds"])
            for row in rows
            if row.get("control_loop_latency_seconds") is not None
            and math.isfinite(float(row["control_loop_latency_seconds"]))
        )
    array = np.asarray(values, dtype=float)
    return {
        "control_cycles": cycles,
        "latency_samples": len(array),
        "mean_control_loop_seconds": float(array.mean()) if len(array) else math.nan,
        "p99_control_loop_seconds": float(np.quantile(array, 0.99)) if len(array) else math.nan,
        "max_control_loop_seconds": float(array.max()) if len(array) else math.nan,
    }


def run_analysis(root: Path, contract_path: Path, output: Path) -> dict[str, Any]:
    contract = read_json(contract_path)
    config = contract["config"]
    config_path = root / contract["config_path"]
    contract_sha = sha256_file(contract_path)
    run_root = resolve_path(config_path, config["paths"]["run_root"])
    runs = select_completed_runs(run_root, contract_sha)
    audit = audit_matrix(runs, config, contract_sha)
    if output.exists():
        raise FileExistsError(f"analysis output is append-only: {output}")
    output.mkdir(parents=True)

    perturbed = aggregate_perturbations(runs, config)
    historical = aggregate_historical(config, config_path)
    paired = pair_with_historical(perturbed, historical, metrics=PAIR_METRICS)
    run_summary = summarize_runs(perturbed)
    delta_summary = summarize_deltas(paired)
    tables = {
        "run_metrics.csv": perturbed,
        "historical_reference_metrics.csv": historical,
        "paired_historical_deltas.csv": paired,
        "condition_summary.csv": run_summary,
        "historical_delta_summary.csv": delta_summary,
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    write_json(output / "integrity_audit.json", audit)
    operational = {
        "schema": "dap.k8s.robustness_operational_summary.v1",
        "comparison_scope": (
            "Absolute perturbation integrity is primary. Historical unperturbed "
            "controls are matched descriptively and are not contemporaneously randomized."
        ),
        "completed_runs": len(runs),
        "profiles": sorted(perturbed.profile.unique()),
        "conditions": sorted(perturbed.condition.unique()),
        "seeds": sorted(int(value) for value in perturbed.seed.unique()),
        "total_perturbation_events": int(perturbed.perturbation_applied_events.sum()),
        **all_action_latency(runs),
        "max_budget_violation_seconds": float(perturbed.budget_violation_seconds.max()),
        "total_deadline_misses": int(perturbed.controller_deadline_misses.sum()),
        "total_monitor_failures": int(perturbed.monitor_failures.sum()),
    }
    write_json(output / "operational_summary.json", operational)

    artifacts = sorted(path for path in output.iterdir() if path.name != "manifest.json")
    manifest = {
        "schema": "dap.k8s.robustness_analysis.v1",
        "status": "completed" if audit["passed"] else "integrity_failed",
        "contract_path": str(contract_path.relative_to(root)),
        "contract_sha256": contract_sha,
        "run_root": str(run_root.relative_to(root)),
        "completed_runs": len(runs),
        "historical_comparison": "descriptive_only_not_contemporaneously_randomized",
        "artifacts": {path.name: sha256_file(path) for path in artifacts},
    }
    write_json(output / "manifest.json", manifest)
    return {"audit": audit, "operational": operational, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = run_analysis(root, args.contract.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["audit"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
