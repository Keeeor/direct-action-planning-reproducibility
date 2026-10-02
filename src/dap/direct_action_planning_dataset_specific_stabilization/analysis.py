from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dap.utils.artifacts import sha256_file, write_json


def assess_dataset_gate(rows: list[dict]) -> dict[str, object]:
    """Apply the frozen first-gate rule to one dataset's evaluation cells."""

    frame = pd.DataFrame(rows)
    required = {
        "dataset",
        "budget",
        "training_seed",
        "method",
        "discounted_return",
        "completion_ratio",
        "slo_violation_rate",
        "total_cost",
        "selected_continuation_weight",
        "budget_overspend",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"gate rows are missing columns: {sorted(missing)}")
    datasets = tuple(frame.dataset.dropna().unique())
    if len(datasets) != 1:
        raise ValueError("gate requires rows from exactly one dataset")
    numeric = frame[list(required - {"dataset", "method"})].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("gate rows contain non-finite values")
    unit = (
        frame.groupby(["dataset", "budget", "training_seed", "method"], as_index=False)[
            [
                "discounted_return",
                "completion_ratio",
                "slo_violation_rate",
                "total_cost",
                "budget_overspend",
                "selected_continuation_weight",
            ]
        ]
        .mean()
    )
    methods = set(unit.method)
    if {"dap_immediate", "dap_calibrated"} - methods:
        raise ValueError("gate requires immediate and calibrated methods")
    immediate = unit[unit.method == "dap_immediate"].set_index(
        ["dataset", "budget", "training_seed"]
    )
    calibrated = unit[unit.method == "dap_calibrated"].set_index(
        ["dataset", "budget", "training_seed"]
    )
    paired = calibrated.join(
        immediate,
        lsuffix="_calibrated",
        rsuffix="_immediate",
        how="inner",
    )
    if len(paired) != 15:
        raise ValueError(f"expected 15 budget-seed cells, got {len(paired)}")
    paired["return_diff"] = (
        paired.discounted_return_calibrated - paired.discounted_return_immediate
    )
    paired["completion_diff"] = (
        paired.completion_ratio_calibrated - paired.completion_ratio_immediate
    )
    paired["slo_diff"] = (
        paired.slo_violation_rate_calibrated
        - paired.slo_violation_rate_immediate
    )
    paired["cost_diff"] = paired.total_cost_calibrated - paired.total_cost_immediate
    seed_effects = paired.groupby(level=[0, 2])[
        ["return_diff", "completion_diff", "slo_diff", "cost_diff"]
    ].mean()
    immediate_return = float(paired.discounted_return_immediate.mean())
    return_gain = float(paired.return_diff.mean())
    relative_gain = return_gain / max(abs(immediate_return), 1.0e-8)
    completion_diff = float(paired.completion_diff.mean())
    slo_diff = float(paired.slo_diff.mean())
    cost_diff = float(paired.cost_diff.mean())
    wins = int((seed_effects.return_diff > 0.0).sum())
    nonzero = int(
        (paired.selected_continuation_weight_calibrated > 0.0).sum()
    )
    overspend = float(
        max(
            paired.budget_overspend_calibrated.max(),
            paired.budget_overspend_immediate.max(),
        )
    )
    cost_guardrail = bool(
        cost_diff <= 0.05 * max(abs(immediate.total_cost.mean()), 1.0e-8)
        or completion_diff >= 0.0
        or slo_diff <= 0.0
    )
    checks = {
        "seed_wins": wins >= 4,
        "relative_gain": relative_gain >= 0.02,
        "nonzero_cells": nonzero >= 10,
        "completion_guardrail": completion_diff >= -0.01,
        "slo_guardrail": slo_diff <= 0.01,
        "cost_guardrail": cost_guardrail,
        "budget_safe": overspend <= 1.0e-8,
    }
    return {
        "dataset": str(datasets[0]),
        "decision": "continue" if all(checks.values()) else "stop",
        "return_gain": return_gain,
        "relative_gain": relative_gain,
        "seed_wins": wins,
        "seed_count": int(len(seed_effects)),
        "nonzero_cells": nonzero,
        "total_cells": int(len(paired)),
        "completion_diff": completion_diff,
        "slo_diff": slo_diff,
        "cost_diff": cost_diff,
        "max_budget_overspend": overspend,
        "checks": checks,
        "paired_cells": paired.reset_index().to_dict(orient="records"),
        "seed_effects": seed_effects.reset_index().to_dict(orient="records"),
    }


def _verify_manifest(run_dir: Path, expected_schema: str) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != expected_schema:
        raise ValueError(f"unexpected run schema: {run_dir}")
    if manifest.get("status") != "completed":
        raise ValueError(f"run is not completed: {run_dir}")
    if manifest.get("formal_test_accessed") is not False:
        raise ValueError(f"formal test access detected: {run_dir}")
    for name, expected in manifest.get("artifacts", {}).items():
        if sha256_file(run_dir / name) != expected:
            raise ValueError(f"artifact hash mismatch: {run_dir / name}")
    return manifest


def analyze_core(
    project_root: str | Path,
    config_path: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    import yaml

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    tier = str(config["tier"])
    result_root = (
        root / "results/direct_action_planning_dataset_specific_stabilization" / tier
    )
    episode_frames: list[pd.DataFrame] = []
    step_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    seen: set[tuple[str, float, int]] = set()
    expected = {
        (str(dataset), float(budget), int(seed))
        for dataset in config["datasets"]
        for budget in config["budgets"]
        for seed in config["seeds"]
    }
    for manifest_path in sorted(result_root.glob("*/**/manifest.json")):
        run_dir = manifest_path.parent
        manifest = _verify_manifest(run_dir, str(config["manifest_schema"]))
        key = (str(manifest["dataset"]), float(manifest["budget"]), int(manifest["training_seed"]))
        if key in seen:
            raise ValueError(f"duplicate core cell: {key}")
        seen.add(key)
        episode_frames.append(pd.read_csv(run_dir / "metrics.csv"))
        step_frames.append(pd.read_csv(run_dir / "steps.csv.gz"))
        selection = pd.read_csv(run_dir / "selection_summary.csv")
        selection["dataset"] = key[0]
        selection["budget"] = key[1]
        selection["training_seed"] = key[2]
        selection_frames.append(selection)
    if seen != expected:
        raise ValueError(f"core grid mismatch: missing={expected - seen}, extra={seen - expected}")
    episodes = pd.concat(episode_frames, ignore_index=True)
    steps = pd.concat(step_frames, ignore_index=True)
    selection = pd.concat(selection_frames, ignore_index=True)
    output = result_root / "analysis_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    episodes.to_csv(output / "episode_metrics.csv", index=False)
    steps.to_csv(output / "step_metrics.csv.gz", index=False, compression="gzip")
    selection.to_csv(output / "selection_summary.csv", index=False)
    unit = episodes.groupby(
        ["dataset", "budget", "training_seed", "method"], as_index=False
    )[
        [
            "discounted_return",
            "completion_ratio",
            "slo_violation_rate",
            "total_cost",
            "budget_overspend",
            "queue_area",
            "decision_ms_mean",
            "decision_ms_p95",
        ]
    ].mean()
    unit.to_csv(output / "unit_metrics.csv", index=False)
    # Gate rows need the selected lambda carried by each episode method.
    gate_unit = unit.merge(
        episodes[
            ["dataset", "budget", "training_seed", "method", "selected_continuation_weight"]
        ].drop_duplicates(),
        on=["dataset", "budget", "training_seed", "method"],
        how="left",
        validate="one_to_one",
    )
    gates = [
        assess_dataset_gate(
            gate_unit[gate_unit.dataset == dataset].to_dict(orient="records")
        )
        for dataset in sorted(gate_unit.dataset.unique())
    ]
    write_json(output / "dataset_gates.json", {row["dataset"]: row for row in gates})
    method_summary = unit.groupby(["dataset", "method"], as_index=False)[
        [
            "discounted_return",
            "completion_ratio",
            "slo_violation_rate",
            "total_cost",
            "budget_overspend",
            "queue_area",
            "decision_ms_mean",
            "decision_ms_p95",
        ]
    ].mean()
    method_summary.to_csv(output / "method_summary.csv", index=False)
    write_json(
        output / "manifest.json",
        {
            "schema": "dap.dap_dataset_specific_stabilization.analysis.v1",
            "status": "completed",
            "formal_test_accessed": False,
            "registered_cells": len(seen),
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "manifest.json"
            },
        },
    )
    return output
