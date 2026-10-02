from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dap.utils.artifacts import sha256_file


METRICS = [
    "mean_Q_star_regret",
    "action_consistency_rate",
    "return_gap_to_paired_optimal",
    "budget_trajectory_mae",
    "completion_rate",
    "slo_violation_rate",
    "total_cost",
]


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _bootstrap_mean_ci(
    values: np.ndarray, seed: int, draws: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))  # type: ignore[return-value]


def paired_comparisons(
    episodes: pd.DataFrame,
    bootstrap_draws: int = 10_000,
    seed: int = 20_260_803,
) -> pd.DataFrame:
    unit = (
        episodes.groupby(["scenario", "model_seed", "method"], as_index=False)[METRICS]
        .mean()
    )
    baseline = unit[unit.method == "frozen_D1"]
    rows: list[dict[str, object]] = []
    candidates = sorted(set(unit.method) - {"frozen_D1"})
    for candidate_index, candidate in enumerate(candidates):
        candidate_frame = unit[unit.method == candidate]
        paired = baseline.merge(
            candidate_frame,
            on=["scenario", "model_seed"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        for metric_index, metric in enumerate(METRICS):
            differences = (
                paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
            ).to_numpy(float)
            if np.allclose(differences, 0.0, atol=1.0e-15):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue
                )
            low, high = _bootstrap_mean_ci(
                differences,
                seed + candidate_index * 101 + metric_index,
                bootstrap_draws,
            )
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": "frozen_D1",
                    "metric": metric,
                    "paired_units": len(differences),
                    "mean_difference": float(np.mean(differences)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "wilcoxon_p": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["bh_q"] = bh_adjust(result.wilcoxon_p.to_numpy(float))
    return result


def pareto_cells(episodes: pd.DataFrame) -> pd.DataFrame:
    cells = (
        episodes.groupby(["scenario", "model_seed", "budget", "method"], as_index=False)
        .agg(
            completion=("completion_rate", "mean"),
            slo=("slo_violation_rate", "mean"),
            cost=("total_cost", "mean"),
        )
    )
    baseline = cells[cells.method == "frozen_D1"]
    candidate = cells[cells.method == "final_context_value"]
    paired = baseline.merge(
        candidate,
        on=["scenario", "model_seed", "budget"],
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
    equal = (service == 0) & (slo == 0) & (cost == 0)
    return pd.DataFrame(
        [
            {
                "candidate": "final_context_value",
                "baseline": "frozen_D1",
                "descriptive_budget_cells": len(paired),
                "candidate_dominates": int(candidate_dominates.sum()),
                "baseline_dominates": int(baseline_dominates.sum()),
                "equal": int(equal.sum()),
                "tradeoff": int((~candidate_dominates & ~baseline_dominates & ~equal).sum()),
            }
        ]
    )


def _method_summary(episodes: pd.DataFrame, runtime: pd.DataFrame) -> pd.DataFrame:
    summary = episodes.groupby("method", as_index=False).agg(
        mean_Q_star_regret=("mean_Q_star_regret", "mean"),
        action_consistency=("action_consistency_rate", "mean"),
        paired_return_gap=("return_gap_to_paired_optimal", "mean"),
        budget_trajectory_mae=("budget_trajectory_mae", "mean"),
        completion_rate=("completion_rate", "mean"),
        slo_violation_rate=("slo_violation_rate", "mean"),
        total_cost=("total_cost", "mean"),
        high_risk_low_cost_accuracy=("high_risk_low_cost_balanced_accuracy", "mean"),
    )
    latency = runtime.groupby("method", as_index=False).decision_latency_ms_mean.mean()
    return summary.merge(latency, on="method", how="left")


def _render_figures(
    validation: pd.DataFrame,
    episodes: pd.DataFrame,
    states: pd.DataFrame,
    output: Path,
) -> list[Path]:
    figure_dir = output / "figures"
    figure_dir.mkdir()
    paths: list[Path] = []
    colors = {"feature": "#0072B2", "gru": "#D55E00"}
    window = (
        validation[validation["mode"].isin(["feature", "gru"])]
        .groupby(["mode", "window"], as_index=False)
        .mean(numeric_only=True)
    )
    fig, axis = plt.subplots(figsize=(6.3, 3.8), constrained_layout=True)
    for mode in ("feature", "gru"):
        subset = window[window["mode"] == mode]
        axis.plot(
            subset.window,
            subset.mean_Q_star_regret,
            marker="o",
            label="History features" if mode == "feature" else "GRU history",
            color=colors[mode],
        )
    axis.set(xlabel="History window L", ylabel="Validation closed-loop Q* regret")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)
    for suffix in ("png", "pdf"):
        path = figure_dir / f"window_sensitivity.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)

    scenario = (
        episodes[
            episodes.method.isin(
                ["frozen_D1", "final_context_value", "oracle_scenario_value"]
            )
        ]
        .groupby(["scenario", "method"], as_index=False)
        .mean(numeric_only=True)
    )
    labels = ["early_burst", "late_burst", "periodic"]
    methods = ["frozen_D1", "final_context_value", "oracle_scenario_value"]
    display = ["Frozen D1", "Context Value", "Oracle context"]
    palette = ["#666666", "#0072B2", "#009E73"]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    width = 0.24
    for offset, (method, name, color) in enumerate(zip(methods, display, palette)):
        values = scenario[scenario.method == method].set_index("scenario").loc[
            labels, "mean_Q_star_regret"
        ]
        axis.bar(x + (offset - 1) * width, values, width, label=name, color=color)
    axis.set_xticks(x, ["Early burst", "Late burst", "Periodic"])
    axis.set_ylabel("Final-test closed-loop Q* regret")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)
    for suffix in ("png", "pdf"):
        path = figure_dir / f"scenario_regret.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)

    state_subset = states[
        states.method.isin(["frozen_D1", "final_context_value"])
        & states.alias_region.astype(str).isin(["False", "True"])
    ]
    state_mean = (
        state_subset.groupby(["method", "alias_region"], as_index=False)
        .mean(numeric_only=True)
    )
    fig, axis = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    x = np.arange(2)
    for offset, (method, name, color) in enumerate(
        [("frozen_D1", "Frozen D1", "#666666"), ("final_context_value", "Context Value", "#CC79A7")]
    ):
        values = [
            float(
                state_mean[
                    (state_mean.method == method)
                    & (state_mean.alias_region.astype(str) == region)
                ].mean_Q_star_regret.iloc[0]
            )
            for region in ("False", "True")
        ]
        axis.bar(x + (offset - 0.5) * 0.34, values, 0.34, label=name, color=color)
    axis.set_xticks(x, ["Non-alias states", "Alias-conflict states"])
    axis.set_ylabel("Full-state Q* regret")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)
    for suffix in ("png", "pdf"):
        path = figure_dir / f"alias_region_regret.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)

    cell = (
        episodes[episodes.method.isin(["frozen_D1", "final_context_value"])]
        .groupby(["scenario", "model_seed", "budget", "method"], as_index=False)
        .mean(numeric_only=True)
    )
    fig, axis = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    for method, name, color, marker in (
        ("frozen_D1", "Frozen D1", "#666666", "o"),
        ("final_context_value", "Context Value", "#E69F00", "s"),
    ):
        subset = cell[cell.method == method]
        axis.scatter(
            subset.total_cost,
            subset.completion_rate,
            s=24,
            alpha=0.75,
            label=name,
            color=color,
            marker=marker,
        )
    axis.set(xlabel="Total resource cost", ylabel="Completion rate")
    axis.legend(frameon=False)
    for suffix in ("png", "pdf"):
        path = figure_dir / f"service_cost_cells.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def run_analysis(project_root: str | Path, run_id: str = "minimal_v1") -> Path:
    root = Path(project_root).resolve()
    source = root / "results/direct_action_planning_context_value" / run_id
    output = source / "analysis_v1"
    if output.exists():
        manifest = output / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "completed":
            return output
        raise RuntimeError(f"append-only analysis output already exists: {output}")
    output.mkdir()
    episodes = pd.read_csv(source / "test_metrics.csv")
    validation = pd.read_csv(source / "validation_metrics.csv")
    states = pd.read_csv(source / "test_state_summary.csv")
    pairs = pd.read_csv(source / "test_pair_ranking.csv")
    runtime = pd.read_csv(source / "runtime.csv")
    gate = json.loads((source / "continuation_gate.json").read_text(encoding="utf-8"))
    method_summary = _method_summary(episodes, runtime)
    comparisons = paired_comparisons(episodes)
    pareto = pareto_cells(episodes)
    scenario_summary = (
        episodes.groupby(["scenario", "method"], as_index=False)[METRICS].mean()
    )
    pair_summary = (
        pairs.groupby(["method", "alias_region"], as_index=False)
        .agg(
            pair_ranking_accuracy=("pair_ranking_accuracy", "mean"),
            action_pairs=("pairs", "sum"),
        )
        if "pairs" in pairs.columns
        else pairs.groupby(["method", "alias_region"], as_index=False).pair_ranking_accuracy.mean()
    )
    method_summary.to_csv(output / "method_summary.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    pareto.to_csv(output / "pareto_cells.csv", index=False)
    scenario_summary.to_csv(output / "scenario_summary.csv", index=False)
    pair_summary.to_csv(output / "pair_ranking_summary.csv", index=False)
    figures = _render_figures(validation, episodes, states, output)
    claims = [
        "# Claim-Evidence Table",
        "",
        "| Claim | Evidence | Status |",
        "|---|---|---|",
        "| State aliasing exists | 6.43% exact-state material action conflicts | Supported in small DP |",
        "| Oracle context repairs LOSO | Zero rollout regret in all three validation scenarios | Supported diagnostically |",
        (
            "| Online history recovers the Oracle gain | "
            f"Recovered {100 * gate['regret']['oracle_gain_recovered']:.2f}% versus 70% gate | Refuted |"
        ),
        (
            "| Context Value passes the minimal gate | "
            f"{gate['cell_wins']}/15 cell wins and alias improvement "
            f"{gate['alias_region']['alias_improvement']:.6f} | Refuted; STOP |"
        ),
        "| Full synthetic/public-trace advantage | Not run after STOP | Not tested |",
    ]
    (output / "claim_evidence_table.md").write_text("\n".join(claims) + "\n", encoding="utf-8")
    inputs = [
        source / "test_metrics.csv",
        source / "test_state_summary.csv",
        source / "test_pair_ranking.csv",
        source / "runtime.csv",
        source / "validation_metrics.csv",
        source / "continuation_gate.json",
    ]
    payload = {
        "schema": "direct_action_planning_context_value.analysis.v1",
        "status": "completed",
        "decision": gate["decision"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "paired_unit": "heldout_scenario_model_seed_after_budget_episode_averaging",
        "paired_units": 15,
        "bootstrap_draws": 10_000,
        "multiple_testing": f"single_BH_family_over_{len(comparisons)}_planned_comparisons",
        "pareto_units": "45 descriptive scenario_model_seed_budget cells",
        "source_sha256": {
            str(path.relative_to(root)): sha256_file(path).removeprefix("sha256:")
            for path in inputs
        },
        "figure_sha256": {
            path.name: sha256_file(path).removeprefix("sha256:") for path in figures
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
