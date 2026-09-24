from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage2_dynamic_budget.direct_action_planning_dataset_specific_stabilization.baseline_analysis import (
    COMPARISON_METRICS,
    DAP_METHOD,
    MEAN_METRICS,
    _bootstrap_summary,
    _comparison_table,
    _internal_ablation,
    _pareto_table,
    _pairing_audit,
    _selection_frequency,
    budget_sensitivity,
    compute_unit_metrics,
)
from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json

from .evaluation import (
    FROZEN_METHODS,
    PRIMARY_METHODS,
    SUPPLEMENTARY_METHODS,
)


METHOD_FAMILIES = {
    method: "primary"
    for method in PRIMARY_METHODS
    if method != DAP_METHOD
}
METHOD_FAMILIES.update({method: "supplementary" for method in SUPPLEMENTARY_METHODS})
METHOD_FAMILIES["dap_immediate"] = "internal_ablation"


def _seed_block_metrics(units: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "discounted_return",
        "completion_ratio",
        "slo_violation_rate",
        "total_cost",
        "queue_area",
        "return_cvar20",
        "completion_p10",
        "slo_p95",
        "budget_overspend_max",
    ]
    return (
        units.groupby(["dataset", "training_seed", "method"], as_index=False)[metrics]
        .mean()
        .sort_values(["dataset", "training_seed", "method"], kind="mergesort")
        .reset_index(drop=True)
    )


def _method_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "discounted_return",
        "completion_ratio",
        "slo_violation_rate",
        "total_cost",
        "queue_area",
        "budget_overspend",
        "decision_ms_mean",
        "decision_ms_p95",
    ]
    rows = []
    for (dataset, method), group in episodes.groupby(["dataset", "method"], sort=True):
        row: dict[str, Any] = {"dataset": dataset, "method": method}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        row["episode_count"] = int(len(group))
        row["seed_count"] = int(group.training_seed.nunique())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "method"], kind="mergesort")


def _test_pairing(episodes: pd.DataFrame) -> dict[str, Any]:
    dap = episodes[episodes.method.eq(DAP_METHOD)].copy()
    baselines = episodes[~episodes.method.eq(DAP_METHOD)].copy()
    return _pairing_audit(dap, baselines)


def build_analysis_tables(
    episodes: pd.DataFrame,
    *,
    method_families: dict[str, str] | None = None,
) -> dict[str, Any]:
    required = {
        "dataset",
        "budget",
        "training_seed",
        "method",
        *MEAN_METRICS,
    }
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"missing analysis columns: {sorted(missing)}")
    observed = set(str(method) for method in episodes.method.unique())
    if observed != set(FROZEN_METHODS) and "dap_calibrated" not in observed:
        raise ValueError("frozen test methods are incomplete")
    units = compute_unit_metrics(episodes)
    families = dict(METHOD_FAMILIES)
    families.update(method_families or {})
    families = {
        method: families.get(method, "unregistered")
        for method in sorted(observed - {DAP_METHOD})
    }
    comparisons = _comparison_table(units, families)
    pareto = _pareto_table(units)
    internal = _internal_ablation(
        units[units.method.isin((DAP_METHOD, "dap_immediate"))].copy()
    )
    return {
        "unit_metrics": units,
        "seed_block_metrics": _seed_block_metrics(units),
        "method_summary": _method_summary(episodes),
        "paired_comparisons": comparisons,
        "pareto_counts": pareto,
        "budget_sensitivity": budget_sensitivity(units),
        "internal_ablation": internal,
        "pairing_integrity": _test_pairing(episodes),
    }


def _action_distribution(steps: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        steps.groupby(["dataset", "method", "action"], as_index=False)
        .size()
        .rename(columns={"size": "action_steps"})
    )
    grouped["action_fraction"] = grouped["action_steps"] / grouped.groupby(
        ["dataset", "method"]
    )["action_steps"].transform("sum")
    return grouped.sort_values(["dataset", "method", "action"], kind="mergesort")


def _test_runtime(episodes: pd.DataFrame) -> pd.DataFrame:
    return (
        episodes.groupby(["dataset", "method"], as_index=False)[
            ["decision_ms_mean", "decision_ms_p95"]
        ]
        .mean()
        .sort_values(["dataset", "method"], kind="mergesort")
    )


def _integrity(episodes: pd.DataFrame, steps: pd.DataFrame) -> dict[str, Any]:
    expected_run_units = 2 * 3 * 5
    expected_episode_rows = expected_run_units * len(FROZEN_METHODS) * 2 * 5
    return {
        "episode_rows": int(len(episodes)),
        "expected_episode_rows": int(expected_episode_rows),
        "episode_rows_complete": bool(len(episodes) == expected_episode_rows),
        "expected_run_units": int(expected_run_units),
        "expected_methods": int(len(FROZEN_METHODS)),
        "test_domains_per_run": 2,
        "episodes_per_domain": 5,
        "step_rows": int(len(steps)),
        "datasets": sorted(str(x) for x in episodes.dataset.unique()),
        "methods": sorted(str(x) for x in episodes.method.unique()),
        "budgets": sorted(float(x) for x in episodes.budget.unique()),
        "training_seeds": sorted(int(x) for x in episodes.training_seed.unique()),
        "all_numeric_finite": bool(
            np.isfinite(episodes.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "max_budget_overspend": float(episodes.budget_overspend.max()),
    }


def _stat_spec(comparisons: pd.DataFrame) -> dict[str, Any]:
    claims = []
    for index, row in comparisons.iterrows():
        claims.append(
            {
                "claim_id": f"TEST-{index:04d}",
                "target": f"dap_calibrated_vs_{row.baseline}",
                "analysis_set": str(row.dataset),
                "metric": str(row.metric),
                "p": float(row.wilcoxon_p),
                "q_fdr": float(row.bh_q),
                "effect_size": float(row.paired_standardized_effect),
                "ci95": [float(row.ci_low), float(row.ci_high)],
                "n": int(row.n_seed_blocks),
                "is_hypothesis": (
                    row.comparison_family == "primary"
                    and row.metric == "discounted_return"
                ),
                "hypothesis_id": (
                    "H4"
                    if row.comparison_family == "primary"
                    and row.metric == "discounted_return"
                    else None
                ),
                "language": "directional_effect_with_guardrails",
                "comparison_family": f"{row.dataset}:{row.comparison_family}:{row.metric}",
            }
        )
    return {
        "project": "direct_action_planning_frozen_temporal_test",
        "claims": claims,
        "correction": "bh",
        "comparisons_run": int(len(comparisons)),
        "comparisons_reported": int(len(comparisons)),
    }


def analyze_frozen_test(
    project_root: str | Path,
    *,
    result_root: str | Path | None = None,
    analysis_name: str = "analysis_v1",
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    results = (
        Path(result_root).resolve()
        if result_root is not None
        else root
        / "results/direct_action_planning_frozen_temporal_test/frozen_model_temporal_test_v1"
    )
    metric_paths = sorted(results.glob("*/*/metrics.csv"))
    step_paths = sorted(results.glob("*/*/steps.csv.gz"))
    if len(metric_paths) != 30 or len(step_paths) != 30:
        raise ValueError("expected 30 completed frozen-test metric and step artifacts")
    episodes = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    steps = pd.concat([pd.read_csv(path) for path in step_paths], ignore_index=True)
    tables = build_analysis_tables(episodes)
    if not analysis_name.startswith("analysis_v") or not analysis_name[10:].isdigit():
        raise ValueError("analysis_name must use the form analysis_vN")
    analysis_dir = results / analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=False)
    episodes.to_csv(analysis_dir / "episode_metrics.csv.gz", index=False, compression="gzip")
    steps.to_csv(analysis_dir / "step_metrics.csv.gz", index=False, compression="gzip")
    for name in (
        "unit_metrics",
        "seed_block_metrics",
        "method_summary",
        "paired_comparisons",
        "pareto_counts",
        "budget_sensitivity",
        "internal_ablation",
    ):
        tables[name].to_csv(analysis_dir / f"{name}.csv", index=False)
    write_json(analysis_dir / "pairing_integrity.json", tables["pairing_integrity"])
    write_json(analysis_dir / "integrity.json", _integrity(episodes, steps))
    _action_distribution(steps).to_csv(analysis_dir / "action_distribution.csv", index=False)
    _test_runtime(episodes).to_csv(analysis_dir / "test_runtime.csv", index=False)
    stat_spec = _stat_spec(tables["paired_comparisons"])
    write_json(analysis_dir / "stat_spec.json", stat_spec)
    source_files = [str(path.relative_to(root)) for path in metric_paths + step_paths]
    write_json(
        analysis_dir / "manifest.json",
        {
            "schema": "stage2.dap_frozen_temporal_test.analysis.v1",
            "status": "complete",
            "analysis_set": "frozen_model_temporal_test_v1",
            "source_count": len(source_files),
            "source_files": source_files,
            "source_sha256": {
                str(path.relative_to(root)): sha256_file(path)
                for path in metric_paths + step_paths
            },
            "statistical_unit": "training_seed_block",
            "comparison_families": sorted(
                tables["paired_comparisons"].comparison_family.unique()
            ),
        },
    )
    return {"analysis_dir": analysis_dir, "stat_spec": analysis_dir / "stat_spec.json"}
