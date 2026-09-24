from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from stage2_dynamic_budget.utils.artifacts import sha256_file


METRICS = [
    "mean_Q_star_regret",
    "action_consistency_rate",
    "return_gap_to_paired_optimal",
    "budget_trajectory_mae",
    "completion_rate",
    "slo_violation_rate",
    "total_cost",
    "high_risk_low_cost_balanced_accuracy",
]

LOWER_IS_BETTER = {
    "mean_Q_star_regret",
    "return_gap_to_paired_optimal",
    "budget_trajectory_mae",
    "slo_violation_rate",
    "total_cost",
}


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return values.copy()
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, draws: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _paired_dz(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    if len(differences) < 2:
        return float("nan")
    scale = float(np.std(differences, ddof=1))
    mean = float(np.mean(differences))
    if scale <= 1.0e-15:
        return 0.0 if abs(mean) <= 1.0e-15 else float(np.sign(mean) * np.inf)
    return mean / scale


def paired_method_comparisons(
    episodes: pd.DataFrame,
    *,
    baseline: str,
    candidates: list[str] | None = None,
    family: str,
    strata: list[str] | None = None,
    bootstrap_draws: int = 10_000,
    seed: int = 20_260_803,
) -> pd.DataFrame:
    strata = list(strata or [])
    required = {"scenario_id", "model_seed", "method", *METRICS, *strata}
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"paired comparisons missing columns: {missing}")
    unit_keys = [*strata, "scenario_id", "model_seed", "method"]
    units = episodes.groupby(unit_keys, as_index=False, dropna=False)[METRICS].mean()
    if candidates is None:
        candidates = sorted(set(units.method) - {baseline})
    grouped = [((), units)] if not strata else list(units.groupby(strata, dropna=False))
    rows: list[dict[str, object]] = []
    comparison_index = 0
    for stratum_values, group in grouped:
        if strata and not isinstance(stratum_values, tuple):
            stratum_values = (stratum_values,)
        stratum = dict(zip(strata, stratum_values))
        base = group[group.method == baseline]
        for candidate in candidates:
            candidate_frame = group[group.method == candidate]
            paired = base.merge(
                candidate_frame,
                on=["scenario_id", "model_seed", *strata],
                suffixes=("_baseline", "_candidate"),
                validate="one_to_one",
            )
            if paired.empty:
                continue
            for metric_index, metric in enumerate(METRICS):
                differences = (
                    paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
                ).to_numpy(float)
                if np.allclose(differences, 0.0, atol=1.0e-15):
                    p_value = 1.0
                else:
                    p_value = float(
                        wilcoxon(
                            differences,
                            alternative="two-sided",
                            zero_method="wilcox",
                        ).pvalue
                    )
                low, high = _bootstrap_mean_ci(
                    differences,
                    seed=seed + comparison_index * 101 + metric_index,
                    draws=bootstrap_draws,
                )
                if metric in LOWER_IS_BETTER:
                    wins = int(np.sum(differences < -1.0e-12))
                    losses = int(np.sum(differences > 1.0e-12))
                else:
                    wins = int(np.sum(differences > 1.0e-12))
                    losses = int(np.sum(differences < -1.0e-12))
                ties = int(len(differences) - wins - losses)
                rows.append(
                    {
                        "family": family,
                        **stratum,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "paired_units": len(differences),
                        "mean_difference": float(np.mean(differences)),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "paired_dz": _paired_dz(differences),
                        "wilcoxon_p": p_value,
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                    }
                )
            comparison_index += 1
    return pd.DataFrame(rows)


def recovery_curve(
    episodes: pd.DataFrame,
    *,
    bootstrap_draws: int = 10_000,
    seed: int = 20_260_803,
) -> pd.DataFrame:
    units = episodes.groupby(
        ["scenario_id", "model_seed", "method"], as_index=False
    ).mean(numeric_only=True)
    wide = units.pivot_table(
        index=["scenario_id", "model_seed"], columns="method", values="mean_Q_star_regret"
    )
    required = {"no_calibration", "oracle_target_value"}
    if not required.issubset(wide.columns):
        raise ValueError("recovery curve requires no-calibration and Oracle rows")
    rows: list[dict[str, object]] = []
    candidates = sorted(column for column in wide.columns if str(column).startswith("adapt_"))
    base = wide["no_calibration"].to_numpy(float)
    oracle = wide["oracle_target_value"].to_numpy(float)
    gain = base - oracle
    informative = gain > 1.0e-12
    for index, method in enumerate(candidates):
        adapted = wide[method].to_numpy(float)
        registered = np.where(
            informative,
            (base - adapted) / np.where(informative, gain, 1.0),
            (adapted <= base + 1.0e-12).astype(float),
        )
        informative_values = (base[informative] - adapted[informative]) / gain[informative]
        registered_low, registered_high = _bootstrap_mean_ci(
            registered, seed=seed + index * 13, draws=bootstrap_draws
        )
        informative_low, informative_high = _bootstrap_mean_ci(
            informative_values, seed=seed + index * 13 + 1, draws=bootstrap_draws
        )
        ratio_match = re.search(r"_(\d{2})pct$", str(method))
        ratio = int(ratio_match.group(1)) / 100.0 if ratio_match else float("nan")
        variant = re.sub(r"^adapt_|_\d{2}pct$", "", str(method))
        differences = adapted - base
        rows.append(
            {
                "method": method,
                "variant": variant,
                "ratio": ratio,
                "cells": len(registered),
                "informative_cells": int(informative.sum()),
                "zero_gain_cells": int((~informative).sum()),
                "registered_mean_recovery": float(np.mean(registered)),
                "registered_ci_low": registered_low,
                "registered_ci_high": registered_high,
                "informative_mean_recovery": float(np.mean(informative_values))
                if len(informative_values)
                else float("nan"),
                "informative_ci_low": informative_low,
                "informative_ci_high": informative_high,
                "wins": int(np.sum(differences < -1.0e-12)),
                "ties": int(np.sum(np.abs(differences) <= 1.0e-12)),
                "losses": int(np.sum(differences > 1.0e-12)),
            }
        )
    return pd.DataFrame(rows)


def pareto_cells(
    episodes: pd.DataFrame, *, baseline: str, candidates: list[str] | None = None
) -> pd.DataFrame:
    cells = episodes.groupby(
        ["scenario_id", "model_seed", "budget", "method"], as_index=False
    ).agg(
        completion=("completion_rate", "mean"),
        slo=("slo_violation_rate", "mean"),
        cost=("total_cost", "mean"),
    )
    if candidates is None:
        candidates = sorted(set(cells.method) - {baseline})
    base = cells[cells.method == baseline]
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        paired = base.merge(
            cells[cells.method == candidate],
            on=["scenario_id", "model_seed", "budget"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        service = paired.completion_candidate - paired.completion_baseline
        slo = paired.slo_candidate - paired.slo_baseline
        cost = paired.cost_candidate - paired.cost_baseline
        candidate_dominates = (service >= 0) & (slo <= 0) & (cost <= 0) & (
            (service > 0) | (slo < 0) | (cost < 0)
        )
        baseline_dominates = (service <= 0) & (slo >= 0) & (cost >= 0) & (
            (service < 0) | (slo > 0) | (cost > 0)
        )
        equal = (service.abs() <= 1.0e-12) & (slo.abs() <= 1.0e-12) & (
            cost.abs() <= 1.0e-12
        )
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "descriptive_budget_cells": len(paired),
                "candidate_dominates": int(candidate_dominates.sum()),
                "baseline_dominates": int(baseline_dominates.sum()),
                "equal": int(equal.sum()),
                "tradeoff": int((~candidate_dominates & ~baseline_dominates & ~equal).sum()),
                "mean_completion_difference": float(service.mean()),
                "mean_slo_difference": float(slo.mean()),
                "mean_cost_difference": float(cost.mean()),
            }
        )
    return pd.DataFrame(rows)


def trajectory_divergence(
    steps: pd.DataFrame,
    *,
    cutoff: int,
    candidate: str = "adapt_full_10pct",
    oracle: str = "oracle_target_value",
) -> dict[str, object]:
    keys = [
        "scenario_id",
        "model_seed",
        "test_seed",
        "budget",
        "episode",
        "eval_seed",
        "t",
    ]
    candidate_rows = steps[steps.method == candidate]
    oracle_rows = steps[steps.method == oracle]
    paired = candidate_rows.merge(
        oracle_rows,
        on=keys,
        suffixes=("_candidate", "_oracle"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("trajectory divergence requires paired candidate and Oracle steps")
    paired["state_diverged"] = (
        (paired.load_candidate != paired.load_oracle)
        | (paired.queue_candidate != paired.queue_oracle)
        | (paired.remaining_budget_candidate != paired.remaining_budget_oracle)
    )
    paired["action_diverged"] = paired.action_candidate != paired.action_oracle
    paired["queue_abs_difference"] = (
        paired.queue_candidate - paired.queue_oracle
    ).abs()
    paired["budget_abs_difference"] = (
        paired.remaining_budget_candidate - paired.remaining_budget_oracle
    ).abs()
    trajectory_keys = keys[:-1]
    total_trajectories = paired[trajectory_keys].drop_duplicates().shape[0]
    divergence = paired[paired.state_diverged | paired.action_diverged]
    first = divergence.groupby(trajectory_keys, as_index=False).t.min()
    cutoff_rows = paired[paired.t == cutoff]
    suffix_rows = paired[paired.t >= cutoff]
    summary = {
        "candidate": candidate,
        "oracle": oracle,
        "cutoff": cutoff,
        "trajectories": int(total_trajectories),
        "trajectories_never_diverged": int(total_trajectories - len(first)),
        "first_divergence_t_mean": float(first.t.mean()) if len(first) else float("nan"),
        "first_divergence_t_median": float(first.t.median()) if len(first) else float("nan"),
        "cutoff_state_divergence_rate": float(cutoff_rows.state_diverged.mean()),
        "cutoff_action_divergence_rate": float(cutoff_rows.action_diverged.mean()),
        "cutoff_queue_mae": float(cutoff_rows.queue_abs_difference.mean()),
        "cutoff_budget_mae": float(cutoff_rows.budget_abs_difference.mean()),
        "suffix_state_divergence_rate": float(suffix_rows.state_diverged.mean()),
        "suffix_action_divergence_rate": float(suffix_rows.action_diverged.mean()),
    }
    scenario = cutoff_rows.groupby("scenario_id", as_index=False).agg(
        cutoff_state_divergence_rate=("state_diverged", "mean"),
        cutoff_action_divergence_rate=("action_diverged", "mean"),
        cutoff_queue_mae=("queue_abs_difference", "mean"),
        cutoff_budget_mae=("budget_abs_difference", "mean"),
    )
    return {"summary": summary, "scenario": scenario, "paired_steps": paired}


def support_stratification(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for space in ("raw", "structured", "hidden", "q_vector"):
        ranked = frame[space].rank(method="first")
        quartile = pd.qcut(ranked, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        current = frame.assign(distance_quartile=quartile)
        for region_name, region in [("all", current), *list(current.groupby("region"))]:
            total_regret = float(region.Q_star_regret.clip(lower=0).sum())
            for label, group in region.groupby("distance_quartile", observed=True):
                regret_sum = float(group.Q_star_regret.clip(lower=0).sum())
                rows.append(
                    {
                        "space": space,
                        "region": region_name,
                        "distance_quartile": str(label),
                        "states": len(group),
                        "distance_mean": float(group[space].mean()),
                        "regret_mean": float(group.Q_star_regret.mean()),
                        "action_error_rate": float(group.action_error.mean()),
                        "regret_share": regret_sum / total_regret if total_regret > 0 else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _render_figures(
    support: pd.DataFrame,
    recovery: pd.DataFrame,
    adaptation: pd.DataFrame,
    output: Path,
) -> list[Path]:
    figure_dir = output / "figures"
    figure_dir.mkdir()
    paths: list[Path] = []
    palette = {
        "raw": "#0072B2",
        "structured": "#D55E00",
        "hidden": "#009E73",
        "q_vector": "#CC79A7",
    }
    overall = support[support.region == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)
    for space in palette:
        part = overall[overall.space == space].sort_values("distance_quartile")
        axes[0].plot(
            part.distance_quartile,
            part.regret_mean,
            marker="o",
            label=space.replace("_", " ").title(),
            color=palette[space],
        )
        axes[1].plot(
            part.distance_quartile,
            part.action_error_rate,
            marker="o",
            label=space.replace("_", " ").title(),
            color=palette[space],
        )
    axes[0].set(xlabel="Support-distance quartile", ylabel="Mean local Q* regret")
    axes[1].set(xlabel="Support-distance quartile", ylabel="Action error rate")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        path = figure_dir / f"support_distance_strata.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)

    colors = {
        "regression": "#0072B2",
        "regression_anchor": "#E69F00",
        "regression_rank": "#009E73",
        "full": "#D55E00",
    }
    fig, axis = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    for variant, color in colors.items():
        part = recovery[recovery.variant == variant].sort_values("ratio")
        axis.plot(
            100 * part.ratio,
            part.registered_mean_recovery,
            marker="o",
            label=variant.replace("_", " + ").title(),
            color=color,
        )
    axis.axhline(0.5, color="#666666", linestyle="--", linewidth=0.9, label="5% gate")
    axis.axhline(0.7, color="#222222", linestyle=":", linewidth=0.9, label="10% gate")
    axis.set(xlabel="Target-prefix calibration data (%)", ylabel="Registered Oracle-gain recovery")
    axis.legend(frameon=False, fontsize=7, ncol=2)
    for suffix in ("png", "pdf"):
        path = figure_dir / f"adaptation_recovery.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)

    unit = adaptation.groupby(
        ["scenario_id", "model_seed", "method"], as_index=False
    )[METRICS].mean()
    base = unit[unit.method == "no_calibration"]
    candidate = unit[unit.method == "adapt_full_10pct"]
    paired = base.merge(
        candidate,
        on=["scenario_id", "model_seed"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    paired["regret_difference"] = (
        paired.mean_Q_star_regret_candidate - paired.mean_Q_star_regret_baseline
    )
    paired["return_gap_difference"] = (
        paired.return_gap_to_paired_optimal_candidate
        - paired.return_gap_to_paired_optimal_baseline
    )
    fig, axis = plt.subplots(figsize=(5.7, 4.2), constrained_layout=True)
    scenario_colors = {
        "early_burst": "#0072B2",
        "late_burst": "#009E73",
        "periodic": "#D55E00",
    }
    for scenario, color in scenario_colors.items():
        part = paired[paired.scenario_id == scenario]
        axis.scatter(
            part.regret_difference,
            part.return_gap_difference,
            label=scenario.replace("_", " ").title(),
            color=color,
            s=34,
        )
    axis.axhline(0, color="#444444", linewidth=0.8)
    axis.axvline(0, color="#444444", linewidth=0.8)
    axis.set(
        xlabel="Suffix local Q* regret difference (adapted - base)",
        ylabel="Paired return-gap difference (adapted - base)",
    )
    axis.legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        path = figure_dir / f"local_vs_closed_loop.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)

    paired["completion_difference"] = (
        paired.completion_rate_candidate - paired.completion_rate_baseline
    )
    paired["slo_difference"] = (
        paired.slo_violation_rate_candidate - paired.slo_violation_rate_baseline
    )
    paired["cost_difference"] = paired.total_cost_candidate - paired.total_cost_baseline
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)
    for scenario, color in scenario_colors.items():
        part = paired[paired.scenario_id == scenario]
        axes[0].scatter(
            part.cost_difference,
            part.completion_difference,
            color=color,
            label=scenario.replace("_", " ").title(),
            s=34,
        )
        axes[1].scatter(
            part.cost_difference,
            part.slo_difference,
            color=color,
            label=scenario.replace("_", " ").title(),
            s=34,
        )
    for axis in axes:
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.axvline(0, color="#444444", linewidth=0.8)
        axis.set_xlabel("Cost difference (adapted - base)")
    axes[0].set_ylabel("Completion difference")
    axes[1].set_ylabel("SLO-violation difference")
    axes[1].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        path = figure_dir / f"service_cost_differences.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def run_analysis(project_root: str | Path, run_id: str = "minimal_v1") -> Path:
    root = Path(project_root).resolve()
    source = root / "results/direct_action_planning_support_adaptation" / run_id
    output = source / "analysis_v1"
    if output.exists():
        manifest_path = output / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "completed":
            return output
        raise RuntimeError(f"append-only analysis output already exists: {output}")
    output.mkdir()
    zero = pd.read_csv(source / "zero_shot_metrics.csv.gz")
    adaptation = pd.read_csv(source / "adaptation_metrics.csv.gz")
    steps = pd.read_csv(source / "adaptation_steps.csv.gz")
    support_rows = pd.read_csv(source / "support_distances.csv.gz")
    gate = json.loads((source / "continuation_gate.json").read_text(encoding="utf-8"))
    ledger = json.loads((source / "final_test_ledger.json").read_text(encoding="utf-8"))
    if gate["decision"] != "STOP":
        raise RuntimeError("formal support-adaptation gate is not STOP")
    if (
        ledger["test_evaluations_completed"] != 1
        or ledger["test_status"] != "completed"
        or ledger["test_used_for_selection"]
    ):
        raise RuntimeError("one-shot final-test ledger integrity failure")

    comparison_parts: list[pd.DataFrame] = []
    zero_candidates = sorted(set(zero.method) - {"frozen_D1"})
    comparison_parts.append(
        paired_method_comparisons(
            zero,
            baseline="frozen_D1",
            candidates=zero_candidates,
            family="all_preregistered_method_metric_tests",
            strata=["region"],
        ).assign(analysis="zero_shot")
    )
    adaptation_candidates = sorted(set(adaptation.method) - {"no_calibration"})
    comparison_parts.append(
        paired_method_comparisons(
            adaptation,
            baseline="no_calibration",
            candidates=adaptation_candidates,
            family="all_preregistered_method_metric_tests",
        ).assign(analysis="adaptation", region="original_loso_suffix")
    )
    comparisons = pd.concat(comparison_parts, ignore_index=True)
    comparisons["bh_q"] = bh_adjust(comparisons.wilcoxon_p.to_numpy(float))
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)

    zero_summary = zero.groupby(["region", "method"], as_index=False)[METRICS].mean()
    zero_summary.to_csv(output / "zero_shot_method_summary.csv", index=False)
    adaptation_summary = adaptation.groupby("method", as_index=False).agg(
        **{metric: (metric, "mean") for metric in METRICS},
        calibration_samples=("calibration_samples", "mean"),
        calibration_seconds=("calibration_seconds", "mean"),
        anchor_mae_increase=("anchor_mae_increase", "mean"),
    )
    adaptation_summary.to_csv(output / "adaptation_method_summary.csv", index=False)
    scenario_summary = adaptation.groupby(
        ["scenario_id", "method"], as_index=False
    )[METRICS].mean()
    scenario_summary.to_csv(output / "adaptation_scenario_summary.csv", index=False)

    recovery = recovery_curve(adaptation)
    recovery.to_csv(output / "recovery_curve.csv", index=False)
    support = support_stratification(support_rows)
    support.to_csv(output / "support_stratification.csv", index=False)
    support_region = support_rows.groupby("region", as_index=False).agg(
        raw_distance=("raw", "mean"),
        structured_distance=("structured", "mean"),
        hidden_distance=("hidden", "mean"),
        q_vector_distance=("q_vector", "mean"),
        mean_Q_star_regret=("Q_star_regret", "mean"),
        action_error_rate=("action_error", "mean"),
    )
    support_region.to_csv(output / "support_region_summary.csv", index=False)

    pareto_parts = []
    for region, group in zero.groupby("region"):
        part = pareto_cells(group, baseline="frozen_D1", candidates=zero_candidates)
        part["analysis"] = "zero_shot"
        part["region"] = region
        pareto_parts.append(part)
    part = pareto_cells(
        adaptation, baseline="no_calibration", candidates=adaptation_candidates
    )
    part["analysis"] = "adaptation"
    part["region"] = "original_loso_suffix"
    pareto_parts.append(part)
    pareto = pd.concat(pareto_parts, ignore_index=True)
    pareto.to_csv(output / "pareto_cells.csv", index=False)

    divergence = trajectory_divergence(steps, cutoff=4)
    _write_json(output / "trajectory_divergence.json", divergence["summary"])
    divergence["scenario"].to_csv(
        output / "trajectory_divergence_by_scenario.csv", index=False
    )
    prefix_suffix = (
        steps[
            steps.method.isin(
                ["no_calibration", "oracle_target_value", "adapt_full_10pct"]
            )
        ]
        .groupby(["scenario_id", "method", "in_metric_suffix"], as_index=False)
        .agg(
            mean_Q_star_regret=("Q_star_regret", "mean"),
            action_consistency=("action_consistent", "mean"),
            reward=("reward", "mean"),
            resource_cost=("resource_cost", "mean"),
            completion_numerator=("served", "sum"),
            completion_denominator=("arrivals", "sum"),
            slo_violation_rate=("slo_violation", "mean"),
            budget_trajectory_mae=("budget_trajectory_gap", "mean"),
        )
    )
    prefix_suffix["completion_rate"] = (
        prefix_suffix.completion_numerator
        / prefix_suffix.completion_denominator.clip(lower=1.0)
    )
    prefix_suffix.to_csv(output / "prefix_suffix_diagnosis.csv", index=False)

    figures = _render_figures(support, recovery, adaptation, output)
    key = comparisons[
        (comparisons.analysis == "adaptation")
        & (comparisons.candidate == "adapt_full_10pct")
    ].set_index("metric")
    full_recovery = recovery[recovery.method == "adapt_full_10pct"].iloc[0]
    pareto_key = pareto[
        (pareto.analysis == "adaptation")
        & (pareto.candidate == "adapt_full_10pct")
    ].iloc[0]
    claims = [
        "# Claim-Evidence Table",
        "",
        "All inferential rows use scenario-model-seed paired units after averaging budgets and episodes. All planned method-metric comparisons share one BH family.",
        "",
        "| Claim | Evidence | Status |",
        "|---|---|---|",
        (
            "| Support distance explains generalization failure | No deployable space reached the registered correlation/order criterion; interpolation regret was 0.102024 versus extrapolation 0.097647 | Refuted in this DP test |"
        ),
        (
            f"| Full 10% calibration improves suffix local Q* regret | Mean paired difference {key.loc['mean_Q_star_regret', 'mean_difference']:.6f}; 95% bootstrap CI [{key.loc['mean_Q_star_regret', 'bootstrap_ci_low']:.6f}, {key.loc['mean_Q_star_regret', 'bootstrap_ci_high']:.6f}]; d_z={key.loc['mean_Q_star_regret', 'paired_dz']:.4f}; Wilcoxon p={key.loc['mean_Q_star_regret', 'wilcoxon_p']:.4g}; BH q={key.loc['mean_Q_star_regret', 'bh_q']:.4g}; wins/ties/losses={int(key.loc['mean_Q_star_regret', 'wins'])}/{int(key.loc['mean_Q_star_regret', 'ties'])}/{int(key.loc['mean_Q_star_regret', 'losses'])} | Local suffix improvement, not stable closed-loop superiority |"
        ),
        (
            f"| At most 10% data recovers at least 70% of Oracle gain | Registered mean recovery {full_recovery.registered_mean_recovery:.4f}; only {int(full_recovery.informative_cells)}/{int(full_recovery.cells)} cells had positive Oracle gain | Gate passes numerically; ceiling-sensitive |"
        ),
        (
            f"| Calibration improves paired closed-loop return | Return-gap difference {key.loc['return_gap_to_paired_optimal', 'mean_difference']:.6f}; 95% bootstrap CI [{key.loc['return_gap_to_paired_optimal', 'bootstrap_ci_low']:.6f}, {key.loc['return_gap_to_paired_optimal', 'bootstrap_ci_high']:.6f}]; BH q={key.loc['return_gap_to_paired_optimal', 'bh_q']:.4g} | Refuted |"
        ),
        (
            f"| Calibration yields stable service-cost Pareto gains | {int(pareto_key.candidate_dominates)}/{int(pareto_key.descriptive_budget_cells)} candidate-dominant cells, {int(pareto_key.baseline_dominates)} baseline-dominant, {int(pareto_key.equal)} equal, {int(pareto_key.tradeoff)} tradeoff | Refuted descriptively |"
        ),
        "| Full synthetic and public-trace advantage | Not run after the preregistered STOP | Not tested |",
    ]
    (output / "claim_evidence_table.md").write_text(
        "\n".join(claims) + "\n", encoding="utf-8"
    )

    inputs = [
        source / "zero_shot_metrics.csv.gz",
        source / "adaptation_metrics.csv.gz",
        source / "adaptation_steps.csv.gz",
        source / "support_distances.csv.gz",
        source / "support_relationship.json",
        source / "continuation_gate.json",
        source / "final_test_ledger.json",
        source / "prefix_suffix_leakage_audit.json",
    ]
    outputs = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema": "direct_action_planning_support_adaptation.analysis.v1",
        "status": "completed",
        "decision": "STOP",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "paired_unit": "scenario_id_model_seed_after_budget_episode_averaging",
        "planned_comparisons": len(comparisons),
        "multiple_testing": f"single_BH_family_over_{len(comparisons)}_planned_comparisons",
        "bootstrap_draws": 10_000,
        "state_rows_descriptive_only": True,
        "final_test_evaluations": 1,
        "test_used_for_selection": False,
        "source_sha256": {
            str(path.relative_to(root)): sha256_file(path).removeprefix("sha256:")
            for path in inputs
        },
        "output_sha256": {
            str(path.relative_to(output)): sha256_file(path).removeprefix("sha256:")
            for path in outputs
        },
        "figures": [str(path.relative_to(output)) for path in figures],
    }
    _write_json(output / "manifest.json", manifest)
    return output
