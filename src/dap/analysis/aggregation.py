from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t

from .statistics import (
    benjamini_hochberg,
    paired_bootstrap,
    paired_effect_size,
    paired_wilcoxon,
)


IDENTIFIERS = ["variant", "method", "budget", "seed", "scenario"]
METRICS = [
    "episode_reward",
    "total_cost",
    "budget_utilization",
    "budget_exhaustion_ratio",
    "mean_latency",
    "p95_latency",
    "slo_violation_rate",
    "completion_rate",
    "mean_queue",
    "risk_budget_correlation",
    "budget_reallocation_ratio",
    "high_risk_budget_mean",
    "low_risk_budget_mean",
]


def collect_runs(run_root: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_frames: list[pd.DataFrame] = []
    runtime_rows: list[dict] = []
    run_directories = sorted(path for path in run_root.glob(f"{prefix}*") if path.is_dir())
    if not run_directories:
        raise FileNotFoundError(f"no run directory matches {prefix!r} below {run_root}")
    for run_dir in run_directories:
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.csv"
        runtime_path = run_dir / "runtime.json"
        if not (manifest_path.exists() and metrics_path.exists() and runtime_path.exists()):
            raise RuntimeError(f"incomplete run bundle: {run_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise RuntimeError(f"run not completed: {run_dir}")
        frame = pd.read_csv(metrics_path)
        frame["run_id"] = run_dir.name
        episode_frames.append(frame)
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        latency_rows = runtime.pop("latency_by_scenario", [])
        latency_mean = float(np.mean([row["decision_latency_ms_mean"] for row in latency_rows]))
        runtime_rows.append(
            {
                "run_id": run_dir.name,
                "method": config["method"],
                "budget": config["budget"],
                "seed": config["seed"],
                "variant": config.get(
                    "variant", f"train_{config.get('training_domain', 'unknown')}"
                ),
                "parameter_count": runtime["parameter_count"],
                "training_seconds": runtime["training_seconds"],
                "decision_latency_ms_mean": latency_mean,
                "device": runtime["device"],
                "global_lambda": runtime["global_lambda"],
            }
        )
    episodes = pd.concat(episode_frames, ignore_index=True)
    duplicated = episodes.duplicated(
        ["variant", "method", "budget", "seed", "scenario", "eval_episode"], keep=False
    )
    if duplicated.any():
        raise RuntimeError("duplicate experimental cells detected")
    return episodes, pd.DataFrame(runtime_rows)


def seed_aggregate(episodes: pd.DataFrame) -> pd.DataFrame:
    selected = [metric for metric in METRICS if metric in episodes.columns]
    return episodes.groupby(IDENTIFIERS, as_index=False, dropna=False)[selected].mean()


def summary_table(seeds: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in METRICS if metric in seeds.columns]
    rows: list[dict] = []
    for keys, group in seeds.groupby(["variant", "method", "budget", "scenario"], dropna=False):
        for metric in metrics:
            values = group[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            if not len(values):
                continue
            mean = float(values.mean())
            standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            half_width = (
                float(t.ppf(0.975, len(values) - 1) * standard_deviation / np.sqrt(len(values)))
                if len(values) > 1
                else np.nan
            )
            rows.append(
                {
                    "variant": keys[0],
                    "method": keys[1],
                    "budget": keys[2],
                    "scenario": keys[3],
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": mean,
                    "std": standard_deviation,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(
    seeds: pd.DataFrame,
    candidate: str,
    baseline: str,
    metrics: Iterable[str] = ("slo_violation_rate", "total_cost", "completion_rate"),
) -> pd.DataFrame:
    rows: list[dict] = []
    index = ["budget", "scenario", "seed"]
    for metric in metrics:
        left = seeds.loc[seeds.method == candidate, index + [metric]].rename(
            columns={metric: "candidate"}
        )
        right = seeds.loc[seeds.method == baseline, index + [metric]].rename(
            columns={metric: "baseline"}
        )
        paired = left.merge(right, on=index, validate="one_to_one")
        for keys, group in paired.groupby(["budget", "scenario"]):
            bootstrap = paired_bootstrap(group.candidate, group.baseline)
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "budget": keys[0],
                    "scenario": keys[1],
                    **bootstrap,
                    "cohens_dz": paired_effect_size(group.candidate, group.baseline),
                    "p_value": paired_wilcoxon(group.candidate, group.baseline),
                }
            )
        overall = paired.groupby("seed", as_index=False)[["candidate", "baseline"]].mean()
        bootstrap = paired_bootstrap(overall.candidate, overall.baseline)
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "metric": metric,
                "budget": "ALL",
                "scenario": "ALL",
                **bootstrap,
                "cohens_dz": paired_effect_size(overall.candidate, overall.baseline),
                "p_value": paired_wilcoxon(overall.candidate, overall.baseline),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = benjamini_hochberg(result.p_value)
    return result


def pareto_table(seeds: pd.DataFrame) -> pd.DataFrame:
    """Method/budget means with a non-dominated flag (lower cost and SLO are better)."""
    table = seeds.groupby(["method", "budget"], as_index=False)[
        ["total_cost", "slo_violation_rate", "completion_rate"]
    ].mean()
    dominated: list[bool] = []
    for _, row in table.iterrows():
        other = table.drop(index=row.name)
        weak = (other.total_cost <= row.total_cost) & (
            other.slo_violation_rate <= row.slo_violation_rate
        )
        strict = (other.total_cost < row.total_cost) | (
            other.slo_violation_rate < row.slo_violation_rate
        )
        dominated.append(bool((weak & strict).any()))
    table["pareto_nondominated"] = ~np.asarray(dominated)
    return table
