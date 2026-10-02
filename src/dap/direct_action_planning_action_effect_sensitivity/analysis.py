"""Paired analysis for the frozen DAP capacity-effect sensitivity family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dap.utils.artifacts import sha256_file, write_json

from .experiment import factor_name, load_protocol


METRICS = ("discounted_return", "completion_ratio", "slo_violation_rate", "total_cost")


def bootstrap_ci(values: np.ndarray, *, seed: int, draws: int = 20_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite and nonempty")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def analyze(project_root: str | Path, config_path: str | Path, output_dir: str | Path) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_protocol(config_path)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"analysis output is append-only: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metrics_frames = []
    step_frames = []
    manifests = []
    for dataset in config["datasets"]:
        for budget in config["budgets"]:
            for seed in config["seeds"]:
                cell = f"{config['tier']}__{dataset}__b{float(budget):.0f}__s{int(seed)}"
                run_dir = root / "results/direct_action_planning_action_effect_sensitivity" / str(config["tier"]) / str(dataset) / cell
                manifest = run_dir / "manifest.json"
                if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")).get("status") != "completed":
                    raise ValueError(f"missing completed sensitivity cell: {run_dir}")
                manifests.append(manifest)
                metrics_frames.append(pd.read_csv(run_dir / "metrics.csv"))
                step_frames.append(pd.read_csv(run_dir / "steps.csv.gz"))
    metrics = pd.concat(metrics_frames, ignore_index=True)
    steps = pd.concat(step_frames, ignore_index=True)
    nominal = factor_name(1.0)
    unit = metrics.groupby(["dataset", "method", "budget", "training_seed"], as_index=False)[list(METRICS) + ["budget_overspend"]].mean()
    keys = ["dataset", "budget", "training_seed"]
    nominal_unit = unit[unit.method == nominal].set_index(keys)
    seed_rows = []
    summary_rows = []
    step_keys = ["dataset", "budget", "training_seed", "domain", "episode", "step"]
    nominal_steps = steps[steps.method == nominal].set_index(step_keys)
    for factor_index, factor in enumerate(config["capacity_factors"]):
        method = factor_name(float(factor))
        if method == nominal:
            continue
        current = unit[unit.method == method].set_index(keys)
        if not current.index.equals(nominal_unit.index):
            raise ValueError(f"unpaired metric units for {method}")
        current_steps = steps[steps.method == method].set_index(step_keys)
        if not current_steps.index.equals(nominal_steps.index):
            raise ValueError(f"unpaired step units for {method}")
        disagreements = (current_steps.action != nominal_steps.action).astype(float)
        disagreement_units = disagreements.groupby(level=[0, 1, 2]).mean()
        for dataset_index, dataset in enumerate(config["datasets"]):
            dataset_key = str(dataset)
            current_dataset = current.loc[dataset_key]
            nominal_dataset = nominal_unit.loc[dataset_key]
            per_seed = pd.DataFrame(index=sorted(int(value) for value in config["seeds"]))
            for metric in METRICS:
                delta = current_dataset[metric] - nominal_dataset[metric]
                per_seed[f"delta_{metric}"] = delta.groupby(level="training_seed").mean()
            current_disagreement = disagreement_units.loc[dataset_key]
            per_seed["action_disagreement_rate"] = current_disagreement.groupby(level="training_seed").mean()
            for seed, row in per_seed.iterrows():
                seed_rows.append({
                    "dataset": dataset_key,
                    "capacity_factor": float(factor),
                    "training_seed": int(seed),
                    **{column: float(value) for column, value in row.items()},
                })
            for metric_index, column in enumerate(per_seed.columns):
                values = per_seed[column].to_numpy(dtype=np.float64)
                low, high = bootstrap_ci(
                    values,
                    seed=2026081200 + factor_index * 100 + dataset_index * 10 + metric_index,
                )
                summary_rows.append({
                    "dataset": dataset_key,
                    "capacity_factor": float(factor),
                    "metric": column,
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "ci_low": low,
                    "ci_high": high,
                })
    pd.DataFrame(seed_rows).to_csv(output / "seed_effects.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "sensitivity_summary.csv", index=False)
    integrity = {
        "expected_cells": len(config["datasets"]) * len(config["budgets"]) * len(config["seeds"]),
        "completed_cells": len(manifests),
        "episode_rows": len(metrics),
        "step_rows": len(steps),
        "max_budget_overspend": float(metrics.budget_overspend.max()),
        "all_factors_present": sorted(metrics.method.unique().tolist()),
    }
    write_json(output / "integrity.json", integrity)
    write_json(output / "manifest.json", {
        "schema": "dap.dap.action_effect_sensitivity.analysis.v1",
        "status": "completed",
        "development_only": True,
        "config_sha256": sha256_file(config_path),
        "input_manifest_sha256": [sha256_file(path) for path in manifests],
        "artifacts": {
            name: sha256_file(output / name)
            for name in ("seed_effects.csv", "sensitivity_summary.csv", "integrity.json")
        },
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = analyze(args.project_root, args.config, args.output)
    print(json.dumps({"status": "completed", "output": str(path)}))


if __name__ == "__main__":
    main()
