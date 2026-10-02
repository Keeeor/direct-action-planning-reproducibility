from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch
import yaml

from dap.utils.artifacts import sha256_file, write_json


DAP_METHOD = "dap_calibrated"
MEAN_METRICS = (
    "discounted_return",
    "completion_ratio",
    "slo_violation_rate",
    "total_cost",
    "queue_area",
    "decision_ms_mean",
    "decision_ms_p95",
    "budget_overspend",
)
COMPARISON_METRICS = (
    "discounted_return",
    "completion_ratio",
    "slo_violation_rate",
    "total_cost",
    "queue_area",
    "return_cvar20",
    "completion_p10",
    "slo_p95",
    "decision_ms_mean",
    "decision_ms_p95",
)
LOWER_IS_BETTER = {
    "slo_violation_rate",
    "total_cost",
    "queue_area",
    "slo_p95",
    "decision_ms_mean",
    "decision_ms_p95",
}


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def compute_unit_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "budget", "training_seed", "method"]
    rows = []
    for key, group in episodes.groupby(keys, sort=True):
        row = dict(zip(keys, key, strict=True))
        row.update({metric: float(group[metric].mean()) for metric in MEAN_METRICS})
        count = max(1, int(np.ceil(0.20 * len(group))))
        row.update(
            {
                "return_cvar20": float(
                    np.sort(group.discounted_return.to_numpy(dtype=np.float64))[:count].mean()
                ),
                "completion_p10": float(group.completion_ratio.quantile(0.10)),
                "slo_p95": float(group.slo_violation_rate.quantile(0.95)),
                "budget_overspend_max": float(group.budget_overspend.max()),
                "episode_count": int(len(group)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys, kind="mergesort").reset_index(drop=True)


def seed_block_difference(
    units: pd.DataFrame,
    *,
    dataset: str,
    baseline: str,
    metric: str,
) -> np.ndarray:
    subset = units[units.dataset.eq(dataset)]
    pivot = subset.pivot(
        index=["budget", "training_seed"], columns="method", values=metric
    )
    differences = pivot[DAP_METHOD if DAP_METHOD in pivot else "dap"] - pivot[baseline]
    return (
        differences.groupby(level="training_seed")
        .mean()
        .sort_index()
        .to_numpy(dtype=np.float64)
    )


def pareto_counts(
    dap_higher: np.ndarray,
    baseline_higher: np.ndarray,
    dap_lower: np.ndarray,
    baseline_lower: np.ndarray,
) -> dict[str, int]:
    dap_higher = np.asarray(dap_higher)
    baseline_higher = np.asarray(baseline_higher)
    dap_lower = np.asarray(dap_lower)
    baseline_lower = np.asarray(baseline_lower)
    dap_dominates = (dap_higher >= baseline_higher) & (dap_lower <= baseline_lower) & (
        (dap_higher > baseline_higher) | (dap_lower < baseline_lower)
    )
    baseline_dominates = (baseline_higher >= dap_higher) & (
        baseline_lower <= dap_lower
    ) & ((baseline_higher > dap_higher) | (baseline_lower < dap_lower))
    ties = np.isclose(dap_higher, baseline_higher) & np.isclose(
        dap_lower, baseline_lower
    )
    return {
        "dap_dominates": int(dap_dominates.sum()),
        "baseline_dominates": int(baseline_dominates.sum()),
        "ties": int(ties.sum()),
        "tradeoffs": int((~dap_dominates & ~baseline_dominates & ~ties).sum()),
    }


def budget_sensitivity(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(units.dataset.unique()):
        subset = units[units.dataset.eq(dataset)]
        for baseline in sorted(set(subset.method) - {DAP_METHOD}):
            for budget in sorted(subset.budget.unique()):
                cell = subset[subset.budget.eq(budget)]
                pivot = cell.pivot(
                    index="training_seed", columns="method"
                )
                return_difference = (
                    pivot[("discounted_return", DAP_METHOD)]
                    - pivot[("discounted_return", baseline)]
                )
                completion_difference = (
                    pivot[("completion_ratio", DAP_METHOD)]
                    - pivot[("completion_ratio", baseline)]
                )
                slo_difference = (
                    pivot[("slo_violation_rate", DAP_METHOD)]
                    - pivot[("slo_violation_rate", baseline)]
                )
                cost_difference = (
                    pivot[("total_cost", DAP_METHOD)]
                    - pivot[("total_cost", baseline)]
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "budget": float(budget),
                        "return_difference": float(return_difference.mean()),
                        "return_wins": int((return_difference > 1.0e-10).sum()),
                        "return_ties": int((np.abs(return_difference) <= 1.0e-10).sum()),
                        "return_losses": int((return_difference < -1.0e-10).sum()),
                        "completion_difference": float(completion_difference.mean()),
                        "slo_difference": float(slo_difference.mean()),
                        "cost_difference": float(cost_difference.mean()),
                    }
                )
    return pd.DataFrame(rows)


def _bootstrap_summary(values: np.ndarray, *, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    p_value = 1.0 if np.allclose(values, 0.0) else float(
        wilcoxon(values, alternative="two-sided").pvalue
    )
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "n_seed_blocks": int(len(values)),
        "mean_favorable_difference": float(values.mean()),
        "median_favorable_difference": float(np.median(values)),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "wilcoxon_p": p_value,
        "paired_standardized_effect": (
            float(values.mean() / standard_deviation)
            if standard_deviation > 1.0e-12
            else 0.0
        ),
    }


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _validate_artifacts(paths: list[Path]) -> list[str]:
    mismatches = []
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("formal_test_accessed") is not False:
            mismatches.append(str(path))
        for name, expected in manifest.get("artifacts", {}).items():
            artifact = path.parent / name
            if not artifact.exists() or sha256_file(artifact) != expected:
                mismatches.append(str(artifact))
    return mismatches


def _pairing_audit(dap: pd.DataFrame, baselines: pd.DataFrame) -> dict:
    keys = [
        "dataset",
        "domain",
        "budget",
        "training_seed",
        "episode",
        "window_seed",
        "window_start",
    ]
    expected = dap[keys].drop_duplicates()
    methods = sorted(baselines.method.unique())
    missing = {}
    for method in methods:
        observed = baselines.loc[baselines.method.eq(method), keys].drop_duplicates()
        joined = expected.merge(observed, on=keys, how="outer", indicator=True)
        counts = joined._merge.value_counts().to_dict()
        missing[method] = {
            "paired": int(counts.get("both", 0)),
            "missing_baseline": int(counts.get("left_only", 0)),
            "extra_baseline": int(counts.get("right_only", 0)),
        }
    passed = all(
        row["missing_baseline"] == 0 and row["extra_baseline"] == 0
        for row in missing.values()
    )
    return {"passed": passed, "pair_keys": keys, "methods": missing}


def _load_episode_metrics(root: Path, baseline_tier: str, dap_tier: str):
    baseline_root = (
        root
        / "results/direct_action_planning_paper_closure"
        / baseline_tier
    )
    dap_root = (
        root
        / "results/direct_action_planning_paper_closure"
        / dap_tier
    )
    baseline_paths = sorted(baseline_root.glob("*/*/metrics.csv"))
    dap_paths = sorted(dap_root.glob("*/*/metrics.csv"))
    if len(baseline_paths) != 30 or len(dap_paths) != 30:
        raise ValueError(
            f"expected 30 baseline and DAP units, found {len(baseline_paths)} and {len(dap_paths)}"
        )
    baselines = pd.concat([pd.read_csv(path) for path in baseline_paths], ignore_index=True)
    dap = pd.concat([pd.read_csv(path) for path in dap_paths], ignore_index=True)
    dap = dap[dap.method.eq(DAP_METHOD)].copy()
    return baseline_root, dap_root, baselines, dap


def _comparison_table(units: pd.DataFrame, families: dict[str, str]) -> pd.DataFrame:
    rows = []
    for dataset in sorted(units.dataset.unique()):
        subset = units[units.dataset.eq(dataset)]
        for baseline in sorted(set(subset.method) - {DAP_METHOD}):
            pivot_index = ["budget", "training_seed"]
            for metric in COMPARISON_METRICS:
                pivot = subset.pivot(index=pivot_index, columns="method", values=metric)
                raw_difference = pivot[DAP_METHOD] - pivot[baseline]
                favorable = -raw_difference if metric in LOWER_IS_BETTER else raw_difference
                seed_values = (
                    favorable.groupby(level="training_seed")
                    .mean()
                    .sort_index()
                    .to_numpy(dtype=np.float64)
                )
                summary = _bootstrap_summary(
                    seed_values,
                    seed=_stable_seed(dataset, baseline, metric),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "comparison_family": families[baseline],
                        "metric": metric,
                        "raw_mean_difference_dap_minus_baseline": float(
                            raw_difference.mean()
                        ),
                        **summary,
                        "unit_wins": int((favorable > 1.0e-10).sum()),
                        "unit_ties": int((np.abs(favorable) <= 1.0e-10).sum()),
                        "unit_losses": int((favorable < -1.0e-10).sum()),
                    }
                )
    comparisons = pd.DataFrame(rows)
    comparisons["bh_q"] = comparisons.groupby(
        ["dataset", "comparison_family", "metric"]
    )["wilcoxon_p"].transform(lambda values: bh_fdr(values.to_numpy()))
    comparisons["evidence_label"] = "not_supportive"
    positive = comparisons.mean_favorable_difference > 0.0
    comparisons.loc[positive, "evidence_label"] = "mixed_positive"
    directional = (
        (comparisons.ci_low > 0.0)
        & (comparisons.unit_wins >= 12)
    )
    comparisons.loc[directional, "evidence_label"] = "consistent_directional"
    confirmed = directional & (comparisons.bh_q < 0.05)
    comparisons.loc[confirmed, "evidence_label"] = "fdr_supported"
    return comparisons


def _pareto_table(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(units.dataset.unique()):
        subset = units[units.dataset.eq(dataset)]
        for baseline in sorted(set(subset.method) - {DAP_METHOD}):
            pivot = subset.pivot(
                index=["budget", "training_seed"], columns="method"
            )
            definitions = {
                "return_cost": ("discounted_return", "total_cost"),
                "completion_cost": ("completion_ratio", "total_cost"),
                "completion_slo": ("completion_ratio", "slo_violation_rate"),
            }
            for name, (higher, lower) in definitions.items():
                counts = pareto_counts(
                    pivot[(higher, DAP_METHOD)].to_numpy(),
                    pivot[(higher, baseline)].to_numpy(),
                    pivot[(lower, DAP_METHOD)].to_numpy(),
                    pivot[(lower, baseline)].to_numpy(),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "relation": name,
                        **counts,
                    }
                )
    return pd.DataFrame(rows)


def _constraint_diagnostics(baseline_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(baseline_root.glob("*/*/training.json")):
        content = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        for method, record in content.items():
            diagnostics = record.get("diagnostics", {})
            history = record.get("history", [])
            rows.append(
                {
                    "dataset": config["dataset"],
                    "budget": float(config["budget"]),
                    "training_seed": int(config["seed"]),
                    "method": method,
                    "training_constraint": record.get("training_constraint"),
                    "final_lambda": float(diagnostics.get("final_lambda", 0.0)),
                    "mean_constraint_cost": float(
                        diagnostics.get("mean_episode_cost", np.nan)
                    ),
                    "projected_step_rate": float(
                        np.mean([entry.get("projected", 0.0) for entry in history])
                        if history
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _cost_analysis(
    baseline_root: Path,
    dap_root: Path,
    units: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for path in sorted(baseline_root.glob("*/*/runtime.json")):
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        runtime = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = torch.load(path.parent / "models.pt", map_location="cpu", weights_only=True)
        for method, seconds in runtime["per_method_training_seconds"].items():
            parameters = int(
                sum(value.numel() for value in checkpoint["models"][method].values())
            )
            rows.append(
                {
                    "dataset": config["dataset"],
                    "budget": float(config["budget"]),
                    "training_seed": int(config["seed"]),
                    "method": method,
                    "training_seconds": float(seconds),
                    "parameter_count": parameters,
                    "process_peak_rss_kib": int(runtime["peak_rss_kib"]),
                }
            )
    for path in sorted(dap_root.glob("*/*/runtime.json")):
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        runtime = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": config["dataset"],
                "budget": float(config["budget"]),
                "training_seed": int(config["seed"]),
                "method": DAP_METHOD,
                "training_seconds": float(runtime["wall_seconds"]),
                "parameter_count": int(
                    runtime["value_parameter_count"]
                    + runtime["forecaster_parameter_count"]
                ),
                "process_peak_rss_kib": int(runtime["peak_rss_kib"]),
            }
        )
    costs = pd.DataFrame(rows)
    latency = units.groupby(["dataset", "method"], as_index=False)[
        ["decision_ms_mean", "decision_ms_p95"]
    ].mean()
    return costs.merge(latency, on=["dataset", "method"], how="left")


def _internal_ablation(dap: pd.DataFrame) -> pd.DataFrame:
    units = compute_unit_metrics(dap)
    rows = []
    controls = sorted(set(units.method) - {DAP_METHOD})
    for dataset in sorted(units.dataset.unique()):
        subset = units[units.dataset.eq(dataset)]
        pivot = subset.pivot(
            index=["budget", "training_seed"], columns="method"
        )
        for control in controls:
            return_difference = (
                pivot[("discounted_return", DAP_METHOD)]
                - pivot[("discounted_return", control)]
            )
            seed_difference = return_difference.groupby(level="training_seed").mean()
            rows.append(
                {
                    "dataset": dataset,
                    "control": control,
                    "return_difference": float(return_difference.mean()),
                    "unit_wins": int((return_difference > 1.0e-10).sum()),
                    "seed_wins": int((seed_difference > 1.0e-10).sum()),
                    "completion_difference": float(
                        (
                            pivot[("completion_ratio", DAP_METHOD)]
                            - pivot[("completion_ratio", control)]
                        ).mean()
                    ),
                    "slo_difference": float(
                        (
                            pivot[("slo_violation_rate", DAP_METHOD)]
                            - pivot[("slo_violation_rate", control)]
                        ).mean()
                    ),
                    "cost_difference": float(
                        (
                            pivot[("total_cost", DAP_METHOD)]
                            - pivot[("total_cost", control)]
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _selection_frequency(dap_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(dap_root.glob("*/*/diagnostics.json")):
        diagnostics = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        selected = diagnostics["selected"]
        rows.append(
            {
                "dataset": config["dataset"],
                "budget": float(config["budget"]),
                "training_seed": int(config["seed"]),
                "selected_iteration": int(selected["candidate_iteration"]),
                "continuation_weight": float(selected["continuation_weight"]),
            }
        )
    raw = pd.DataFrame(rows)
    return (
        raw.groupby(
            ["dataset", "budget", "selected_iteration", "continuation_weight"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "selected_cells"})
    )


def _repair_history(root: Path) -> pd.DataFrame:
    rows = []
    result_root = root / "results/direct_action_planning_paper_closure"
    for version in range(1, 5):
        path = (
            result_root
            / f"paper_closure_core_v{version}"
            / "analysis_v1/dataset_gates.json"
        )
        if not path.exists():
            continue
        content = json.loads(path.read_text(encoding="utf-8"))
        for dataset, result in content.items():
            rows.append(
                {
                    "version": f"v{version}",
                    "dataset": dataset,
                    "decision": result["decision"],
                    "return_gain_vs_immediate": float(result["return_gain"]),
                    "relative_gain": float(result["relative_gain"]),
                    "seed_wins": int(result["seed_wins"]),
                    "nonzero_cells": int(result["nonzero_cells"]),
                    "completion_difference": float(result["completion_diff"]),
                    "slo_difference": float(result["slo_diff"]),
                    "cost_difference": float(result["cost_diff"]),
                }
            )
    return pd.DataFrame(rows)


def _write_claim_tables(
    output: Path,
    comparisons: pd.DataFrame,
    pareto: pd.DataFrame,
    summaries: pd.DataFrame,
    ablations: pd.DataFrame,
    budget_results: pd.DataFrame,
) -> None:
    returns = comparisons[comparisons.metric.eq("discounted_return")]
    lines = [
        "# Claim-Evidence Table",
        "",
        "Development-only evidence. Deltas are DAP minus baseline; five training seeds are the ",
        "independent blocks, while budgets and windows are repeated measures. Wilcoxon p-values are ",
        "two-sided and BH-FDR is applied within dataset and pre-frozen comparison family.",
        "",
        "| Dataset | Family | Baseline | Return delta | Wins | 95% seed-block CI | BH q | Completion delta | SLO delta | Cost delta | Evidence |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in returns.sort_values(["dataset", "comparison_family", "baseline"]).itertuples():
        subset = comparisons[
            comparisons.dataset.eq(row.dataset)
            & comparisons.baseline.eq(row.baseline)
        ].set_index("metric")
        lines.append(
            "| {dataset} | {family} | {baseline} | {delta:+.3f} | {wins}/15 | "
            "[{low:+.3f}, {high:+.3f}] | {q:.4f} | {completion:+.4f} | {slo:+.4f} | "
            "{cost:+.2f} | {evidence} |".format(
                dataset=row.dataset,
                family=row.comparison_family,
                baseline=row.baseline,
                delta=row.raw_mean_difference_dap_minus_baseline,
                wins=row.unit_wins,
                low=row.ci_low,
                high=row.ci_high,
                q=row.bh_q,
                completion=subset.loc[
                    "completion_ratio", "raw_mean_difference_dap_minus_baseline"
                ],
                slo=subset.loc[
                    "slo_violation_rate", "raw_mean_difference_dap_minus_baseline"
                ],
                cost=subset.loc[
                    "total_cost", "raw_mean_difference_dap_minus_baseline"
                ],
                evidence=row.evidence_label,
            )
        )
    (output / "claim_evidence_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = [
        "# Dataset-Specific Baseline Analysis",
        "",
        "Status: development-only exploratory evidence; formal test arrays were not loaded.",
        "",
        "## Scope",
        "",
        "Azure2019 and GenTD26 are trained and interpreted separately. This analysis does not make a ",
        "zero-shot or universal scheduling claim. Traditional analytic controllers are excluded from ",
        "the primary learned-method ranking.",
        "",
        "## Main result",
        "",
    ]
    for dataset in sorted(summaries.dataset.unique()):
        dap = summaries[
            summaries.dataset.eq(dataset) & summaries.method.eq(DAP_METHOD)
        ].iloc[0]
        dataset_returns = returns[returns.dataset.eq(dataset)]
        report.append(
            f"- {dataset}: DAP mean return {dap.discounted_return:.3f}; return gains were positive "
            f"against {int((dataset_returns.mean_favorable_difference > 0).sum())}/"
            f"{len(dataset_returns)} learning baselines."
        )
    report.extend(
        [
            "",
            "With only five independent seed blocks, the minimum attainable two-sided Wilcoxon p-value ",
            "is 0.0625, so no comparison can be described as conventionally significant after BH-FDR. ",
            "The appropriate claim is consistent development-set direction where the bootstrap interval ",
            "excludes zero and unit wins are broad, not confirmatory superiority.",
            "",
            "## Pareto interpretation",
            "",
        ]
    )
    for row in pareto[pareto.relation.eq("completion_cost")].itertuples():
        report.append(
            f"- {row.dataset} vs {row.baseline}: DAP completion-cost dominance in "
            f"{row.dap_dominates}/15 units; baseline dominance in {row.baseline_dominates}/15; "
            f"tradeoffs in {row.tradeoffs}/15."
        )
    report.extend(["", "## Innovation attribution", ""])
    for row in ablations[
        ablations.control.eq("dap_immediate")
    ].sort_values("dataset").itertuples():
        report.append(
            f"- {row.dataset}: learned continuation versus immediate branch planning changed return "
            f"by {row.return_difference:+.3f} ({row.seed_wins}/5 seed blocks favorable); contribution "
            f"must be classified from the preregistered gate in repair_history.csv, not inferred from "
            f"external-baseline wins."
        )
    report.extend(["", "## Budget sensitivity", ""])
    for dataset in sorted(budget_results.dataset.unique()):
        primary = budget_results[
            budget_results.dataset.eq(dataset)
            & budget_results.baseline.isin(
                returns[
                    returns.dataset.eq(dataset)
                    & returns.comparison_family.eq("primary")
                ].baseline
            )
        ]
        counts = primary.groupby("budget").return_wins.sum()
        report.append(
            f"- {dataset}: aggregate primary-comparator seed wins by budget were "
            + ", ".join(f"{budget:g}: {int(value)}/30" for budget, value in counts.items())
            + "."
        )
    (output / "FINAL_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_stat_spec(
    output: Path,
    comparisons: pd.DataFrame,
    units: pd.DataFrame,
) -> None:
    claims = []
    returns = comparisons[comparisons.metric.eq("discounted_return")]
    for row in returns.sort_values(["dataset", "comparison_family", "baseline"]).itertuples():
        raw_seed_values = seed_block_difference(
            units,
            dataset=row.dataset,
            baseline=row.baseline,
            metric="discounted_return",
        )
        claims.append(
            {
                "claim_id": f"return:{row.dataset}:dap_vs_{row.baseline}",
                "text": (
                    "development-set directional return comparison; no statistically "
                    "significant superiority asserted"
                ),
                "p": float(row.wilcoxon_p),
                "q_fdr": float(row.bh_q),
                "effect_size": float(row.paired_standardized_effect),
                "effect_kind": "paired_standardized_mean_difference",
                "ci95": [float(row.ci_low), float(row.ci_high)],
                "n": int(row.n_seed_blocks),
                "asserted_grade": "none",
                "is_hypothesis": False,
                "seeds": [float(value) for value in raw_seed_values],
            }
        )
    write_json(
        output / "stat_spec.json",
        {
            "project": "direct-action-planning-dataset-specific-stabilization",
            "correction": "bh",
            "comparisons_run": len(claims),
            "comparisons_reported": len(claims),
            "claims": claims,
        },
    )


def run_baseline_analysis(
    project_root: str | Path,
    *,
    baseline_tier: str = "dataset_specific_baselines_core_v1",
    dap_tier: str = "paper_closure_core_v1",
    config_path: str | Path,
    analysis_name: str = "analysis_v1",
) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    methods = [str(value) for value in config["methods"]]
    families = {
        method: (
            "primary"
            if method in set(config["primary_methods"])
            else "supplementary"
        )
        for method in methods
    }
    baseline_root, dap_root, baselines, dap = _load_episode_metrics(
        root, baseline_tier, dap_tier
    )
    output = baseline_root / analysis_name
    if output.exists():
        raise FileExistsError(f"analysis is append-only: {output}")
    output.mkdir(parents=True)

    baseline_manifests = sorted(baseline_root.glob("*/*/manifest.json"))
    dap_manifests = sorted(dap_root.glob("*/*/manifest.json"))
    baseline_failures = sorted(baseline_root.glob("**/failure.json"))
    mismatches = _validate_artifacts(baseline_manifests + dap_manifests)
    integrity = {
        "baseline_manifest_count": len(baseline_manifests),
        "dap_manifest_count": len(dap_manifests),
        "baseline_failure_count": len(baseline_failures),
        "artifact_mismatches": mismatches,
        "formal_test_accessed": False,
        "passed": (
            len(baseline_manifests) == 30
            and len(dap_manifests) == 30
            and not baseline_failures
            and not mismatches
        ),
    }
    write_json(output / "integrity.json", integrity)
    if not integrity["passed"]:
        raise ValueError("input integrity gate failed")

    pairing = _pairing_audit(dap, baselines)
    write_json(output / "pairing_integrity.json", pairing)
    if not pairing["passed"]:
        raise ValueError("episode pairing gate failed")

    episodes = pd.concat([dap, baselines], ignore_index=True)
    units = compute_unit_metrics(episodes)
    summaries = units.groupby(["dataset", "method"], as_index=False)[
        list(MEAN_METRICS) + ["return_cvar20", "completion_p10", "slo_p95"]
    ].mean()
    seed_blocks = units.groupby(
        ["dataset", "training_seed", "method"], as_index=False
    )[list(COMPARISON_METRICS)].mean()
    comparisons = _comparison_table(units, families)
    pareto = _pareto_table(units)
    diagnostics = _constraint_diagnostics(baseline_root)
    costs = _cost_analysis(baseline_root, dap_root, units)
    budget_results = budget_sensitivity(units)
    ablations = _internal_ablation(
        pd.concat(
            [pd.read_csv(path) for path in sorted(dap_root.glob("*/*/metrics.csv"))],
            ignore_index=True,
        )
    )
    selection = _selection_frequency(dap_root)
    repair_history = _repair_history(root)

    episodes.to_csv(output / "episode_metrics.csv.gz", index=False, compression="gzip")
    units.to_csv(output / "unit_metrics.csv", index=False)
    seed_blocks.to_csv(output / "seed_block_metrics.csv", index=False)
    summaries.to_csv(output / "method_summary.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    pareto.to_csv(output / "pareto_counts.csv", index=False)
    diagnostics.to_csv(output / "constraint_diagnostics.csv", index=False)
    costs.to_csv(output / "cost_analysis.csv", index=False)
    budget_results.to_csv(output / "budget_sensitivity.csv", index=False)
    ablations.to_csv(output / "internal_ablation.csv", index=False)
    selection.to_csv(output / "selection_frequency.csv", index=False)
    repair_history.to_csv(output / "repair_history.csv", index=False)
    _write_claim_tables(
        output,
        comparisons,
        pareto,
        summaries,
        ablations,
        budget_results,
    )
    _write_stat_spec(output, comparisons, units)

    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(
        output / "manifest.json",
        {
            "schema": "dap.dap_dataset_specific_stabilization.analysis.v1",
            "baseline_tier": baseline_tier,
            "dap_tier": dap_tier,
            "config_sha256": sha256_file(config_path),
            "analysis_code_sha256": sha256_file(Path(__file__)),
            "analysis_runner_sha256": sha256_file(
                root / "scripts/analyze_direct_action_planning_dataset_specific_baselines.py"
            ),
            "development_only": True,
            "formal_test_accessed": False,
            "artifacts": artifacts,
        },
    )
    return output
