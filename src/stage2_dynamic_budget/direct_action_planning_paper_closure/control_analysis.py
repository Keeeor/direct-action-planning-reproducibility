"""Pre-registered, all-cell analysis for same-information DAP controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json

from .analysis import LOWER_IS_BETTER, _bh, _bootstrap, _pareto


FULL_METHOD = "dap_full"
METRICS = (
    "discounted_return",
    "completion_ratio",
    "slo_violation_rate",
    "total_cost",
    "queue_area",
    "final_queue",
    "decision_ms_mean",
    "decision_ms_p95",
)


def _expected_cells(config: dict[str, Any]) -> set[tuple[str, float, int]]:
    return {
        (str(dataset), float(budget), int(seed))
        for dataset in config["datasets"]
        for budget in config["budgets"]
        for seed in config["seeds"]
    }


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _validate_artifacts(run_dir: Path, manifest: dict[str, Any]) -> None:
    for name, expected in manifest.get("artifacts", {}).items():
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"artifact integrity failure: {path}")


def _load_controls(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_root = root / "results/direct_action_planning_paper_closure" / str(config["tier"])
    expected = _expected_cells(config)
    seen: set[tuple[str, float, int]] = set()
    episodes: list[pd.DataFrame] = []
    steps: list[pd.DataFrame] = []
    for manifest_path in sorted(result_root.glob("*/**/manifest.json")):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("formal_test_accessed") is not False:
            raise ValueError(f"invalid development control manifest: {run_dir}")
        key = (str(manifest["dataset"]), float(manifest["budget"]), int(manifest["training_seed"]))
        if key in seen:
            raise ValueError(f"duplicate control cell: {key}")
        seen.add(key)
        _validate_artifacts(run_dir, manifest)
        episodes.append(pd.read_csv(run_dir / "metrics.csv"))
        steps.append(pd.read_csv(run_dir / "steps.csv.gz"))
    if seen != expected:
        raise ValueError(f"control grid mismatch: missing={expected-seen}, extra={seen-expected}")
    return pd.concat(episodes, ignore_index=True), pd.concat(steps, ignore_index=True)


def _action_behavior(steps: pd.DataFrame) -> pd.DataFrame:
    index = [
        "dataset", "domain", "budget", "training_seed", "seed", "episode",
        "window_seed", "window_start", "step",
    ]
    full = steps[steps.method.eq(FULL_METHOD)][index + ["action", "remaining_budget"]]
    full = full.rename(columns={"action": "full_action", "remaining_budget": "full_remaining_budget"})
    rows: list[dict[str, Any]] = []
    for method in sorted(set(steps.method) - {FULL_METHOD}):
        other = steps[steps.method.eq(method)][index + ["action", "remaining_budget"]]
        merged = full.merge(other, on=index, how="inner", validate="one_to_one")
        if len(merged) != len(full):
            raise ValueError(f"step alignment failed for {method}")
        per_seed = merged.groupby(["dataset", "budget", "training_seed"], as_index=False).agg(
            action_agreement=("action", lambda values: float((values.to_numpy() == merged.loc[values.index, "full_action"].to_numpy()).mean())),
            remaining_budget_mae=("remaining_budget", lambda values: float(np.abs(values.to_numpy() - merged.loc[values.index, "full_remaining_budget"].to_numpy()).mean())),
        )
        per_seed["control"] = method
        rows.extend(per_seed.to_dict("records"))
    return pd.DataFrame(rows)


def control_analysis(
    project_root: str | Path,
    config_path: str | Path,
    *,
    analysis_name: str = "analysis_v1",
) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    episodes, steps = _load_controls(root, config)
    output = root / "results/direct_action_planning_paper_closure" / str(config["tier"]) / analysis_name
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    episodes.to_csv(output / "episode_metrics.csv", index=False)
    steps.to_csv(output / "step_metrics.csv.gz", index=False, compression="gzip")

    unit = episodes.groupby(["dataset", "budget", "training_seed", "method"], as_index=False)[list(METRICS)].mean()
    unit.to_csv(output / "unit_metrics.csv", index=False)
    methods = tuple(str(value) for value in config["methods"])
    if FULL_METHOD not in methods:
        raise ValueError("control protocol is missing the full DAP reference")

    comparisons: list[dict[str, Any]] = []
    for dataset in sorted(unit.dataset.unique()):
        subset = unit[unit.dataset.eq(dataset)]
        for control in sorted(set(methods) - {FULL_METHOD}):
            for metric in METRICS:
                pivot = subset.pivot(index=["budget", "training_seed"], columns="method", values=metric)
                raw = pivot[FULL_METHOD] - pivot[control]
                favorable = -raw if metric in LOWER_IS_BETTER else raw
                per_seed = favorable.groupby(level="training_seed").mean().to_numpy()
                comparisons.append({
                    "dataset": dataset,
                    "control": control,
                    "metric": metric,
                    "raw_mean_full_minus_control": float(raw.mean()),
                    **_bootstrap(per_seed, _stable_seed(dataset, control, metric)),
                })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["bh_q_within_dataset_metric"] = comparison_frame.groupby(
        ["dataset", "metric"], sort=False
    )["wilcoxon_p"].transform(lambda values: _bh(values.to_numpy()))
    comparison_frame.to_csv(output / "paired_control_comparisons.csv", index=False)

    pareto: list[dict[str, Any]] = []
    for dataset in sorted(unit.dataset.unique()):
        subset = unit[unit.dataset.eq(dataset)]
        pivot = subset.pivot(index=["budget", "training_seed"], columns="method")
        for control in sorted(set(methods) - {FULL_METHOD}):
            for (budget, seed), row in pivot.iterrows():
                pareto.append({
                    "dataset": dataset,
                    "budget": float(budget),
                    "training_seed": int(seed),
                    "control": control,
                    "relation_return_cost": _pareto(
                        row[("discounted_return", FULL_METHOD)], row[("total_cost", FULL_METHOD)],
                        row[("discounted_return", control)], row[("total_cost", control)],
                    ),
                    "relation_completion_cost": _pareto(
                        row[("completion_ratio", FULL_METHOD)], row[("total_cost", FULL_METHOD)],
                        row[("completion_ratio", control)], row[("total_cost", control)],
                    ),
                })
    pareto_frame = pd.DataFrame(pareto)
    pareto_frame.to_csv(output / "pareto_cells.csv", index=False)
    pareto_frame.groupby(["dataset", "control", "relation_return_cost"], as_index=False).size().rename(
        columns={"size": "cells"}
    ).to_csv(output / "pareto_summary.csv", index=False)

    behavior = _action_behavior(steps)
    behavior.to_csv(output / "action_budget_behavior.csv", index=False)
    behavior.groupby(["dataset", "control"], as_index=False).agg(
        action_agreement_mean=("action_agreement", "mean"),
        remaining_budget_mae_mean=("remaining_budget_mae", "mean"),
    ).to_csv(output / "action_budget_behavior_summary.csv", index=False)
    unit.groupby(["dataset", "method"], as_index=False)[list(METRICS)].mean().to_csv(
        output / "method_summary.csv", index=False
    )
    write_json(output / "manifest.json", {
        "schema": "stage2.dap_paper_closure.controls_analysis.v1",
        "status": "completed",
        "config_sha256": sha256_file(config_path),
        "registered_cells": len(_expected_cells(config)),
        "artifacts": {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    })
    return output
