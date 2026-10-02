"""Seed-level paired analysis for the locked DAP--PDS/ADP comparison."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import yaml

from dap.utils.artifacts import sha256_file, write_json


METHODS = ("dap_calibrated", "pds_adp")
METRICS = (
    "discounted_return", "completion_ratio", "slo_violation_rate", "total_cost"
)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p_values must be a finite one-dimensional array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    if len(values) == 0:
        return values.copy()
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def build_seed_units(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "budget", "training_seed", "method", *METRICS}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics lack required columns: {sorted(missing)}")
    frame = metrics[metrics.method.isin(METHODS)].copy()
    if set(frame.method.unique()) != set(METHODS):
        raise ValueError("both registered methods are required")
    numeric = [*METRICS]
    for optional in ("decision_ms_mean", "decision_ms_p95"):
        if optional in frame:
            numeric.append(optional)
    units = (
        frame.groupby(["dataset", "budget", "training_seed", "method"], as_index=False)[numeric]
        .mean()
        .sort_values(["dataset", "budget", "training_seed", "method"])
        .reset_index(drop=True)
    )
    counts = units.groupby(["dataset", "budget", "training_seed"]).method.nunique()
    if not (counts == len(METHODS)).all():
        raise ValueError("unpaired seed-level method units")
    return units


def _exact_signed_rank(difference: np.ndarray) -> tuple[float, float, float]:
    nonzero = np.asarray(difference, dtype=np.float64)
    nonzero = nonzero[np.abs(nonzero) > 1.0e-12]
    if len(nonzero) == 0:
        return 0.0, 1.0, 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    observed_signed = float(np.sum(np.sign(nonzero) * ranks))
    total = float(np.sum(ranks))
    extreme = 0
    combinations = 1 << len(ranks)
    for mask in range(combinations):
        signs = np.fromiter(
            (1.0 if mask & (1 << index) else -1.0 for index in range(len(ranks))),
            dtype=np.float64,
        )
        if abs(float(np.sum(signs * ranks))) >= abs(observed_signed) - 1.0e-12:
            extreme += 1
    p_value = float(extreme / combinations)
    rank_biserial = observed_signed / total
    w_plus = float(np.sum(ranks[nonzero > 0.0]))
    w_minus = float(np.sum(ranks[nonzero < 0.0]))
    return min(w_plus, w_minus), p_value, rank_biserial


def _bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, samples: int = 10_000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_comparison(
    units: pd.DataFrame, *, metric: str, bootstrap_seed: int, bootstrap_samples: int = 10_000
) -> pd.DataFrame:
    if metric not in units:
        raise ValueError(f"unknown metric: {metric}")
    rows: list[dict[str, Any]] = []
    for offset, ((dataset, budget), group) in enumerate(
        units.groupby(["dataset", "budget"], sort=True)
    ):
        pivot = group.pivot(index="training_seed", columns="method", values=metric)
        if set(pivot.columns) != set(METHODS) or pivot.isna().any().any():
            raise ValueError(f"unpaired method observations for {dataset}/{budget}/{metric}")
        difference = (
            pivot["dap_calibrated"] - pivot["pds_adp"]
        ).to_numpy(dtype=np.float64)
        statistic, p_value, effect = _exact_signed_rank(difference)
        ci_low, ci_high = _bootstrap_mean_ci(
            difference, seed=int(bootstrap_seed) + 1009 * offset,
            samples=int(bootstrap_samples),
        )
        rows.append(
            {
                "dataset": dataset,
                "budget": float(budget),
                "metric": metric,
                "difference_definition": "dap_minus_pds",
                "n": int(len(difference)),
                "mean_difference_dap_minus_pds": float(np.mean(difference)),
                "std_difference": float(np.std(difference, ddof=1)) if len(difference) > 1 else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial": effect,
                "dap_better_count": int(np.sum(difference > 1.0e-12)) if metric in {"discounted_return", "completion_ratio"} else int(np.sum(difference < -1.0e-12)),
                "pds_better_count": int(np.sum(difference < -1.0e-12)) if metric in {"discounted_return", "completion_ratio"} else int(np.sum(difference > 1.0e-12)),
                "tie_count": int(np.sum(np.abs(difference) <= 1.0e-12)),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value_bh_within_metric"] = benjamini_hochberg(result.p_value.to_numpy())
    return result


def classify_pareto(
    *, dap_completion: float, dap_slo: float, dap_cost: float,
    pds_completion: float, pds_slo: float, pds_cost: float,
    tolerance: float = 1.0e-12,
) -> str:
    dap_no_worse = (
        dap_completion >= pds_completion - tolerance
        and dap_slo <= pds_slo + tolerance
        and dap_cost <= pds_cost + tolerance
    )
    pds_no_worse = (
        pds_completion >= dap_completion - tolerance
        and pds_slo <= dap_slo + tolerance
        and pds_cost <= dap_cost + tolerance
    )
    dap_strict = (
        dap_completion > pds_completion + tolerance
        or dap_slo < pds_slo - tolerance
        or dap_cost < pds_cost - tolerance
    )
    pds_strict = (
        pds_completion > dap_completion + tolerance
        or pds_slo < dap_slo - tolerance
        or pds_cost < dap_cost - tolerance
    )
    if dap_no_worse and dap_strict:
        return "dap_dominates"
    if pds_no_worse and pds_strict:
        return "pds_dominates"
    if dap_no_worse and pds_no_worse:
        return "tie"
    return "tradeoff"


def _load_runs(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[pd.DataFrame] = []
    steps: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    seen: set[tuple[str, float, int]] = set()
    tier = str(config["tier"])
    base = root / "results/direct_action_planning_pds_adp" / tier
    for run_dir in sorted(base.glob("*/*")):
        failure_path = run_dir / "failure.json"
        if failure_path.exists():
            failures.append({"run_dir": str(run_dir.relative_to(root)), **json.loads(failure_path.read_text())})
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            failures.append({"run_dir": str(run_dir.relative_to(root)), "status": "missing_manifest"})
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "completed":
            failures.append({"run_dir": str(run_dir.relative_to(root)), "status": "incomplete"})
            continue
        cell = (str(manifest["dataset"]), float(manifest["budget"]), int(manifest["training_seed"]))
        if cell in seen:
            raise ValueError(f"duplicate completed temporal cell: {cell}")
        seen.add(cell)
        for name, expected_hash in manifest["artifacts"].items():
            artifact = run_dir / name
            if not artifact.exists() or sha256_file(artifact) != expected_hash:
                raise ValueError(f"artifact hash mismatch: {artifact}")
        metrics.append(pd.read_csv(run_dir / "metrics.csv"))
        steps.append(pd.read_csv(run_dir / "steps.csv.gz"))
    expected = {
        (str(dataset), float(budget), int(seed))
        for dataset, budget, seed in itertools.product(
            config["datasets"], config["budgets"], config["seeds"]
        )
    }
    if seen != expected:
        raise ValueError(f"temporal grid mismatch: missing={expected-seen}, extra={seen-expected}")
    return pd.concat(metrics, ignore_index=True), pd.concat(steps, ignore_index=True), pd.DataFrame(failures)


def _method_summary(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, budget, method), group in units.groupby(["dataset", "budget", "method"], sort=True):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            digest = hashlib.sha256(f"{dataset}/{budget}/{method}/{metric}".encode()).digest()
            seed = int.from_bytes(digest[:4], "big")
            low, high = _bootstrap_mean_ci(values, seed=seed)
            rows.append({
                "dataset": dataset, "budget": float(budget), "method": method,
                "metric": metric, "n": len(values), "mean": float(values.mean()),
                "std": float(values.std(ddof=1)), "ci95_low": low, "ci95_high": high,
            })
    return pd.DataFrame(rows)


def _pareto(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, budget, seed), group in units.groupby(["dataset", "budget", "training_seed"], sort=True):
        by_method = group.set_index("method")
        dap = by_method.loc["dap_calibrated"]
        pds = by_method.loc["pds_adp"]
        rows.append({
            "dataset": dataset, "budget": float(budget), "training_seed": int(seed),
            "classification": classify_pareto(
                dap_completion=float(dap.completion_ratio), dap_slo=float(dap.slo_violation_rate), dap_cost=float(dap.total_cost),
                pds_completion=float(pds.completion_ratio), pds_slo=float(pds.slo_violation_rate), pds_cost=float(pds.total_cost),
            ),
        })
    return pd.DataFrame(rows)


def _action_disagreement(steps: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "domain", "budget", "training_seed", "episode", "step"]
    action = steps.pivot(index=keys, columns="method", values="action")
    cost = steps.pivot(index=keys, columns="method", values="cost")
    if action.isna().any().any() or cost.isna().any().any():
        raise ValueError("unpaired temporal steps")
    paired = action.rename(columns={name: f"action_{name}" for name in METHODS}).join(
        cost.rename(columns={name: f"cost_{name}" for name in METHODS})
    ).reset_index()
    paired["action_disagreement"] = paired.action_dap_calibrated != paired.action_pds_adp
    paired["cost_difference_dap_minus_pds"] = paired.cost_dap_calibrated - paired.cost_pds_adp
    return (
        paired.groupby(["dataset", "budget", "training_seed"], as_index=False)
        .agg(
            steps=("action_disagreement", "size"),
            action_disagreement_rate=("action_disagreement", "mean"),
            mean_step_cost_difference_dap_minus_pds=("cost_difference_dap_minus_pds", "mean"),
        )
    )


def analyze_temporal_results(
    project_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"analysis output is append-only: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metrics, steps, failures = _load_runs(root, config)
    units = build_seed_units(metrics)
    paired = pd.concat(
        [paired_comparison(units, metric=metric, bootstrap_seed=202608090 + offset * 100_003)
         for offset, metric in enumerate(METRICS)],
        ignore_index=True,
    )
    summary = _method_summary(units)
    pareto = _pareto(units)
    pareto_counts = (
        pareto.groupby(["dataset", "budget", "classification"], as_index=False)
        .size().rename(columns={"size": "seed_count"})
    )
    disagreement = _action_disagreement(steps)
    runtime = (
        steps.groupby(["dataset", "budget", "training_seed", "method"], as_index=False)
        .decision_ms.agg(["mean", "median", lambda series: series.quantile(0.95), "max"])
        .reset_index().rename(columns={"<lambda_0>": "p95"})
    )
    files = {
        "unit_metrics.csv": units,
        "method_summary.csv": summary,
        "paired_comparisons.csv": paired,
        "pareto_seed.csv": pareto,
        "pareto_counts.csv": pareto_counts,
        "action_disagreement.csv": disagreement,
        "decision_runtime.csv": runtime,
        "all_episode_metrics.csv": metrics,
        "failures.csv": failures,
    }
    for name, frame in files.items():
        frame.to_csv(output / name, index=False)
    write_json(output / "manifest.json", {
        "schema": "dap.dap_pds_adp.analysis.v1",
        "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "statistical_unit": "trained seed after averaging domains and episodes",
        "primary_family": "discounted_return across 10 dataset-budget cells",
        "multiple_testing": "BH-FDR within each metric family",
        "paired_test": "two-sided exact signed-rank sign enumeration",
        "bootstrap_samples": 10_000,
        "completed_cells": int(units.groupby(["dataset", "budget", "training_seed"]).ngroups),
        "failed_runs": int(len(failures)),
        "artifacts": {name: sha256_file(output / name) for name in files},
    })
    return output
