from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json


DAP = "dap_calibrated"
PRIMARY_BASELINES = (
    "double_dqn",
    "ppo",
    "ppo_lagrangian",
    "cpo",
    "p3o",
    "budgeted_fitted_q",
    "lcpo",
)
METRICS = {
    "discounted_return": "higher",
    "completion_ratio": "higher",
    "slo_violation_rate": "lower",
    "total_cost": "lower",
}


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    rows.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def classify_pareto(
    dap_return: float,
    dap_cost: float,
    baseline_return: float,
    baseline_cost: float,
) -> str:
    dap_weak = dap_return >= baseline_return and dap_cost <= baseline_cost
    base_weak = baseline_return >= dap_return and baseline_cost <= dap_cost
    if dap_weak and (dap_return > baseline_return or dap_cost < baseline_cost):
        return "dap_dominates"
    if base_weak and (baseline_return > dap_return or baseline_cost < dap_cost):
        return "baseline_dominates"
    return "tradeoff"


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int = 20260805) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_effect(
    frame: pd.DataFrame,
    *,
    metric: str,
    baseline: str,
    favorable: str,
) -> dict[str, Any]:
    subset = frame[frame.method.isin((DAP, baseline))]
    wide = subset.pivot(index="training_seed", columns="method", values=metric).dropna()
    if list(wide.index) != sorted(wide.index) or len(wide) < 2:
        wide = wide.sort_index()
    raw = wide[DAP].to_numpy() - wide[baseline].to_numpy()
    effect = raw if favorable == "higher" else -raw
    try:
        test = stats.wilcoxon(effect, alternative="two-sided", zero_method="wilcox")
        p_value = float(test.pvalue)
    except ValueError:
        p_value = 1.0
    lower, upper = _bootstrap_mean_ci(effect)
    standard_deviation = float(np.std(effect, ddof=1))
    dz = float(np.mean(effect) / standard_deviation) if standard_deviation > 0.0 else 0.0
    return {
        "baseline": baseline,
        "metric": metric,
        "favorable": favorable,
        "n": int(len(effect)),
        "mean_effect": float(np.mean(effect)),
        "ci95_low": lower,
        "ci95_high": upper,
        "wilcoxon_p": p_value,
        "cohens_dz": dz,
        "wins": int(np.sum(effect > 0.0)),
        "ties": int(np.sum(np.isclose(effect, 0.0))),
        "losses": int(np.sum(effect < 0.0)),
    }


def _load_lcpo_test(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    base = (
        root
        / "results/direct_action_planning_recent_sota/recent_sota_lcpo_temporal_test_v1_locked"
    )
    metric_frames: list[pd.DataFrame] = []
    step_frames: list[pd.DataFrame] = []
    sources: list[Path] = []
    failures = list(base.glob("*/*/failure.json"))
    if failures:
        raise ValueError(f"LCPO temporal failures exist: {len(failures)}")
    for manifest in sorted(base.glob("*/*/manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("status") != "completed" or data.get("formal_test_accessed") is not True:
            raise ValueError(f"invalid LCPO temporal manifest: {manifest}")
        directory = manifest.parent
        metric_frames.append(pd.read_csv(directory / "metrics.csv"))
        step_frames.append(pd.read_csv(directory / "steps.csv.gz"))
        sources.extend((manifest, directory / "metrics.csv", directory / "steps.csv.gz"))
    if len(metric_frames) != 100:
        raise ValueError(f"expected 100 LCPO temporal cells, found {len(metric_frames)}")
    return pd.concat(metric_frames, ignore_index=True), pd.concat(step_frames, ignore_index=True), sources


def _unit_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "discounted_return",
        "completion_ratio",
        "slo_violation_rate",
        "total_cost",
        "budget_overspend",
        "queue_area",
        "decision_ms_mean",
        "decision_ms_p95",
    ]
    return (
        episodes.groupby(["dataset", "method", "budget", "training_seed"], as_index=False)[numeric]
        .mean()
        .sort_values(["dataset", "method", "budget", "training_seed"])
    )


def _method_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "discounted_return",
        "completion_ratio",
        "slo_violation_rate",
        "total_cost",
        "budget_overspend",
        "queue_area",
        "decision_ms_mean",
        "decision_ms_p95",
    ]
    return episodes.groupby(["dataset", "method"], as_index=False)[numeric].mean()


def _pareto(unit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for dataset in sorted(unit.dataset.unique()):
        dap = unit[(unit.dataset == dataset) & (unit.method == DAP)].set_index(
            ["budget", "training_seed"]
        )
        for baseline in PRIMARY_BASELINES:
            other = unit[(unit.dataset == dataset) & (unit.method == baseline)].set_index(
                ["budget", "training_seed"]
            )
            for key in dap.index.intersection(other.index):
                drow = dap.loc[key]
                brow = other.loc[key]
                rows.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "budget": float(key[0]),
                        "training_seed": int(key[1]),
                        "classification": classify_pareto(
                            float(drow.discounted_return),
                            float(drow.total_cost),
                            float(brow.discounted_return),
                            float(brow.total_cost),
                        ),
                    }
                )
    cells = pd.DataFrame(rows)
    summary = (
        cells.groupby(["dataset", "baseline", "classification"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ("dap_dominates", "baseline_dominates", "tradeoff"):
        if column not in summary:
            summary[column] = 0
    return cells, summary


def _cost_matched(unit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in sorted(unit.dataset.unique()):
        for baseline in PRIMARY_BASELINES:
            for training_seed in sorted(unit.training_seed.unique()):
                dap = unit[
                    (unit.dataset == dataset)
                    & (unit.method == DAP)
                    & (unit.training_seed == training_seed)
                ]
                other = unit[
                    (unit.dataset == dataset)
                    & (unit.method == baseline)
                    & (unit.training_seed == training_seed)
                ]
                curve = (
                    other.groupby("total_cost", as_index=False)[
                        ["discounted_return", "slo_violation_rate"]
                    ]
                    .mean()
                    .sort_values("total_cost")
                )
                if len(curve) < 2:
                    continue
                costs = curve.total_cost.to_numpy()
                for row in dap.itertuples():
                    if not costs[0] <= row.total_cost <= costs[-1]:
                        continue
                    matched_return = float(
                        np.interp(row.total_cost, costs, curve.discounted_return)
                    )
                    matched_slo = float(
                        np.interp(row.total_cost, costs, curve.slo_violation_rate)
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "baseline": baseline,
                            "training_seed": int(training_seed),
                            "budget": float(row.budget),
                            "dap_cost": float(row.total_cost),
                            "return_difference": float(row.discounted_return - matched_return),
                            "slo_difference": float(row.slo_violation_rate - matched_slo),
                        }
                    )
    return pd.DataFrame(rows)


def _training_costs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lcpo = root / "results/direct_action_planning_recent_sota/recent_sota_lcpo_frontier_v1_locked"
    for path in sorted(lcpo.glob("*/*/training.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": config["dataset"],
                "budget": config["budget"],
                "training_seed": config["seed"],
                "method": "lcpo",
                "training_seconds": data["elapsed_seconds"],
            }
        )
    baseline = root / "results/direct_action_planning_paper_closure/paper_closure_baselines_v3_locked"
    for path in sorted(baseline.glob("*/*/training.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        for method, result in data.items():
            rows.append(
                {
                    "dataset": config["dataset"],
                    "budget": config["budget"],
                    "training_seed": config["seed"],
                    "method": method,
                    "training_seconds": result["elapsed_seconds"],
                }
            )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["dataset", "method"], as_index=False)
        .training_seconds.agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )


def _evidence_grade(row: pd.Series) -> tuple[str, str]:
    if row.q_fdr < 0.05 and row.ci95_low > 0.0:
        return "weak", "within this post-hoc locked evaluation, DAP improved the metric relative to LCPO"
    if row.q_fdr < 0.05 and row.ci95_high < 0.0:
        return "weak", "within this post-hoc locked evaluation, DAP worsened the metric relative to LCPO"
    return "none", "no corrected difference was detected in this post-hoc comparison"


def analyze(project_root: str | Path, output_dir: str | Path) -> Path:
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    old_analysis = (
        root
        / "results/direct_action_planning_paper_closure/paper_closure_temporal_test_v2_locked/analysis_v1"
    )
    old_episodes_path = old_analysis / "episode_metrics.csv"
    old_steps_path = old_analysis / "step_metrics.csv.gz"
    old_episodes = pd.read_csv(old_episodes_path)
    old_steps = pd.read_csv(old_steps_path)
    lcpo_episodes, lcpo_steps, lcpo_sources = _load_lcpo_test(root)
    episodes = pd.concat((old_episodes, lcpo_episodes), ignore_index=True)
    steps = pd.concat((old_steps, lcpo_steps), ignore_index=True)
    episodes.to_csv(output / "episode_metrics.csv", index=False)
    steps.to_csv(output / "step_metrics.csv.gz", index=False, compression="gzip")
    unit = _unit_metrics(episodes)
    summary = _method_summary(episodes)
    unit.to_csv(output / "unit_metrics.csv", index=False)
    summary.to_csv(output / "method_summary.csv", index=False)

    lcpo_comparisons: list[dict[str, Any]] = []
    for dataset in sorted(unit.dataset.unique()):
        dataset_unit = unit[unit.dataset == dataset].groupby(
            ["method", "training_seed"], as_index=False
        )[list(METRICS)].mean()
        family_rows: list[dict[str, Any]] = []
        for metric, favorable in METRICS.items():
            result = paired_effect(
                dataset_unit, metric=metric, baseline="lcpo", favorable=favorable
            )
            result.update(
                {
                    "dataset": dataset,
                    "comparison": "dap_calibrated_minus_lcpo_favorable_orientation",
                    "family_id": f"posthoc-lcpo-{dataset}",
                }
            )
            family_rows.append(result)
        q_values = benjamini_hochberg(
            np.asarray([row["wilcoxon_p"] for row in family_rows])
        )
        for row, q_value in zip(family_rows, q_values, strict=True):
            row["q_fdr"] = float(q_value)
            lcpo_comparisons.append(row)
    comparisons = pd.DataFrame(lcpo_comparisons)
    comparisons.to_csv(output / "lcpo_comparisons.csv", index=False)

    pareto_cells, pareto_summary = _pareto(unit)
    pareto_cells.to_csv(output / "pareto_cells.csv", index=False)
    pareto_summary.to_csv(output / "pareto_summary.csv", index=False)
    matched = _cost_matched(unit)
    matched.to_csv(output / "matched_frontier.csv", index=False)
    training = _training_costs(root)
    training.to_csv(output / "training_cost_summary.csv", index=False)

    evidence_claims: list[dict[str, Any]] = []
    for index, row in comparisons.iterrows():
        grade, language = _evidence_grade(row)
        evidence_claims.append(
            {
                "claim_id": f"recent-lcpo-{index + 1:02d}",
                "dataset": row.dataset,
                "metric": row.metric,
                "comparison": row.comparison,
                "effect_favorable_to_dap": float(row.mean_effect),
                "ci95": [float(row.ci95_low), float(row.ci95_high)],
                "p": float(row.wilcoxon_p),
                "q_fdr": float(row.q_fdr),
                "effect_size_dz": float(row.cohens_dz),
                "n_training_seeds": int(row.n),
                "grade": grade,
                "language_ceiling": language,
                "post_hoc_extension": True,
            }
        )
    evidence = {
        "schema": "light.evidence_strength.v1",
        "project": "stage2_dynamic_budget",
        "analysis": "direct_action_planning_recent_sota",
        "claims": evidence_claims,
    }
    write_json(output / "evidence_strength.json", evidence)

    lcpo_summary = summary[summary.method.isin((DAP, "lcpo"))]
    lines = [
        "# Recent-SOTA LCPO Extension Analysis",
        "",
        "This analysis is a disclosed post-hoc extension on historically accessed test partitions.",
        "Effects below are oriented so positive values favor DAP.",
        "",
        "## Aggregate metrics",
        "",
        _markdown_table(lcpo_summary),
        "",
        "## Paired DAP versus LCPO comparisons",
        "",
        _markdown_table(comparisons),
        "",
        "## Return-cost Pareto cells",
        "",
        _markdown_table(pareto_summary[pareto_summary.baseline == "lcpo"]),
        "",
        "No universal or best-known-performance claim follows from this post-hoc comparison.",
    ]
    (output / "ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    source_paths = [old_episodes_path, old_steps_path, *lcpo_sources]
    digest = hashlib.sha256()
    for path in sorted(source_paths):
        digest.update(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n".encode())
    write_json(
        output / "manifest.json",
        {
            "schema": "stage2.dap_recent_sota.analysis.v1",
            "status": "completed",
            "post_hoc_extension": True,
            "episode_rows": int(len(episodes)),
            "step_rows": int(len(steps)),
            "methods": sorted(episodes.method.unique()),
            "max_budget_overspend": float(episodes.budget_overspend.max()),
            "source_inventory_sha256": "sha256:" + digest.hexdigest(),
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "manifest.json"
            },
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.project_root, args.output))


if __name__ == "__main__":
    main()
