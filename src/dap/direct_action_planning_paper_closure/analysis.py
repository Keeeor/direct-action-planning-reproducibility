"""Pre-registered paper-closure analysis for temporal run bundles."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dap.utils.artifacts import sha256_file, write_json

from .temporal_evaluation import METHODS


LOWER_IS_BETTER = {"slo_violation_rate", "total_cost", "budget_overspend", "decision_ms_mean", "decision_ms_p95"}
PRIMARY_BASELINES = ("double_dqn", "ppo", "ppo_lagrangian", "cpo", "p3o", "budgeted_fitted_q")


def _bootstrap(values: np.ndarray, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite and non-empty")
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    p = 1.0 if np.allclose(values, 0.0) else float(wilcoxon(values).pvalue)
    return {
        "n_seed_blocks": int(len(values)),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(sample, 0.025)),
        "ci_high": float(np.quantile(sample, 0.975)),
        "wilcoxon_p": p,
        "positive_seed_count": int((values > 0.0).sum()),
        "negative_seed_count": int((values < 0.0).sum()),
    }


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _pareto(a_service, a_cost, b_service, b_cost) -> str:
    if a_service >= b_service and a_cost <= b_cost and (a_service > b_service or a_cost < b_cost):
        return "dap_dominates"
    if b_service >= a_service and b_cost <= a_cost and (b_service > a_service or b_cost < a_cost):
        return "baseline_dominates"
    if np.isclose(a_service, b_service) and np.isclose(a_cost, b_cost):
        return "tie"
    return "tradeoff"


def _interior_interp(x: np.ndarray, y: np.ndarray, target: float) -> float | None:
    order = np.argsort(x, kind="mergesort")
    x = np.asarray(x, dtype=np.float64)[order]
    y = np.asarray(y, dtype=np.float64)[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) < 2 or target < unique_x[0] or target > unique_x[-1]:
        return None
    unique_y = np.asarray([y[inverse == index].mean() for index in range(len(unique_x))])
    return float(np.interp(target, unique_x, unique_y))


def _load_temporal(root: Path, tier: str, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_root = root / "results/direct_action_planning_paper_closure" / tier
    expected = {
        (str(dataset), float(budget), int(seed))
        for dataset in config["datasets"]
        for budget in config["budgets"]
        for seed in config["seeds"]
    }
    seen: set[tuple[str, float, int]] = set()
    episodes: list[pd.DataFrame] = []
    steps: list[pd.DataFrame] = []
    for manifest_path in sorted(result_root.glob("*/**/manifest.json")):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or not manifest.get("current_checkpoint_test_accessed"):
            raise ValueError(f"invalid temporal manifest: {run_dir}")
        key = (str(manifest["dataset"]), float(manifest["budget"]), int(manifest["training_seed"]))
        if key in seen:
            raise ValueError(f"duplicate temporal cell: {key}")
        seen.add(key)
        episodes.append(pd.read_csv(run_dir / "metrics.csv"))
        steps.append(pd.read_csv(run_dir / "steps.csv.gz"))
    if seen != expected:
        raise ValueError(f"temporal grid mismatch: missing={expected-seen}, extra={seen-expected}")
    return pd.concat(episodes, ignore_index=True), pd.concat(steps, ignore_index=True)


def frontier_analysis(root: str | Path, contract_path: str | Path, *, analysis_name: str = "analysis_v1") -> Path:
    root = Path(root).resolve()
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    config = contract["config"]
    episodes, steps = _load_temporal(root, str(config["tier"]), config)
    output = root / "results/direct_action_planning_paper_closure" / str(config["tier"]) / analysis_name
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    episodes.to_csv(output / "episode_metrics.csv", index=False)
    steps.to_csv(output / "step_metrics.csv.gz", index=False, compression="gzip")
    metrics = ["discounted_return", "completion_ratio", "slo_violation_rate", "total_cost", "budget_overspend", "decision_ms_mean", "decision_ms_p95"]
    unit = episodes.groupby(["dataset", "budget", "training_seed", "method"], as_index=False)[metrics].mean()
    unit.to_csv(output / "unit_metrics.csv", index=False)

    comparisons: list[dict[str, Any]] = []
    pareto: list[dict[str, Any]] = []
    for dataset in sorted(unit.dataset.unique()):
        subset = unit[unit.dataset.eq(dataset)]
        for baseline in PRIMARY_BASELINES:
            for metric in ("discounted_return", "completion_ratio", "slo_violation_rate", "total_cost"):
                pivot = subset.pivot(index=["budget", "training_seed"], columns="method", values=metric)
                if baseline not in pivot or "dap_calibrated" not in pivot:
                    raise ValueError(f"missing method {baseline} or dap_calibrated")
                raw = pivot["dap_calibrated"] - pivot[baseline]
                favorable = -raw if metric in LOWER_IS_BETTER else raw
                seed_values = favorable.groupby(level="training_seed").mean().to_numpy()
                comparisons.append({"dataset": dataset, "baseline": baseline, "metric": metric, "raw_mean_dap_minus_baseline": float(raw.mean()), **_bootstrap(seed_values, _stable_seed(dataset, baseline, metric))})
            piv = subset.pivot(index=["budget", "training_seed"], columns="method")
            for index, row in piv.iterrows():
                pareto.append({"dataset": dataset, "budget": float(index[0]), "training_seed": int(index[1]), "baseline": baseline, "relation_return_cost": _pareto(row[("discounted_return", "dap_calibrated")], row[("total_cost", "dap_calibrated")], row[("discounted_return", baseline)], row[("total_cost", baseline)]), "relation_completion_cost": _pareto(row[("completion_ratio", "dap_calibrated")], row[("total_cost", "dap_calibrated")], row[("completion_ratio", baseline)], row[("total_cost", baseline)])})

    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["bh_q_within_dataset_metric"] = comparison_frame.groupby(["dataset", "metric"])["wilcoxon_p"].transform(lambda values: _bh(values.to_numpy()))
    comparison_frame.to_csv(output / "paired_comparisons.csv", index=False)
    pareto_frame = pd.DataFrame(pareto)
    pareto_frame.to_csv(output / "pareto_cells.csv", index=False)
    pareto_summary = pareto_frame.groupby(["dataset", "baseline", "relation_return_cost"], as_index=False).size().rename(columns={"size": "cells"})
    pareto_summary.to_csv(output / "pareto_summary.csv", index=False)

    matched: list[dict[str, Any]] = []
    for dataset in sorted(unit.dataset.unique()):
        subset = unit[unit.dataset.eq(dataset)]
        for baseline in PRIMARY_BASELINES:
            for seed in sorted(subset.training_seed.unique()):
                dap = subset[(subset.method == "dap_calibrated") & (subset.training_seed == seed)]
                base = subset[(subset.method == baseline) & (subset.training_seed == seed)]
                for _, dap_row in dap.iterrows():
                    return_at_cost = _interior_interp(base.total_cost.to_numpy(), base.discounted_return.to_numpy(), float(dap_row.total_cost))
                    slo_at_cost = _interior_interp(base.total_cost.to_numpy(), base.slo_violation_rate.to_numpy(), float(dap_row.total_cost))
                    cost_at_slo = _interior_interp(base.slo_violation_rate.to_numpy(), base.total_cost.to_numpy(), float(dap_row.slo_violation_rate))
                    matched.append({"dataset": dataset, "baseline": baseline, "training_seed": int(seed), "dap_budget": float(dap_row.budget), "dap_cost": float(dap_row.total_cost), "dap_return": float(dap_row.discounted_return), "dap_slo": float(dap_row.slo_violation_rate), "baseline_return_at_dap_cost": return_at_cost, "baseline_slo_at_dap_cost": slo_at_cost, "baseline_cost_at_dap_slo": cost_at_slo, "cost_match_interior": return_at_cost is not None, "slo_match_interior": cost_at_slo is not None})
    matched_frame = pd.DataFrame(matched)
    matched_frame.to_csv(output / "matched_frontier.csv", index=False)

    utilization = unit.assign(budget_utilization=unit["total_cost"] / unit["budget"].clip(lower=1.0))
    utilization.to_csv(output / "budget_utilization.csv", index=False)

    method_summary = unit.groupby(["dataset", "method"], as_index=False)[metrics + ["budget"]].mean()
    method_summary.to_csv(output / "method_summary.csv", index=False)
    step_summary = steps.groupby(["dataset", "method"], as_index=False)[["decision_ms"]].agg(["mean", "median", "max"]).reset_index()
    step_summary.to_csv(output / "decision_time.csv", index=False)
    write_json(output / "manifest.json", {"schema": "dap.dap_paper_closure.analysis.v1", "status": "completed", "contract_sha256": sha256_file(Path(contract_path)), "registered_cells": len(unit[["dataset", "budget", "training_seed"]].drop_duplicates()), "artifacts": {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"}})
    return output
