"""Programmatic, evidence-bound figures for the locked DAP closure study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json


plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
})


COLORS = {
    "dap_calibrated": "#0072B2",
    "double_dqn": "#E69F00",
    "cpo": "#009E73",
    "ppo_lagrangian": "#CC79A7",
    "budgeted_fitted_q": "#D55E00",
    "dap_full": "#0072B2",
    "dap_immediate_structured": "#E69F00",
    "mpc_4": "#009E73",
    "mpc_8": "#CC79A7",
    "dap_black_box_transition": "#D55E00",
    "dap_actor_distilled": "#56B4E9",
    "dap_no_budget_horizon": "#999999",
}
LABELS = {
    "dap_calibrated": "Full DAP",
    "double_dqn": "Double DQN",
    "cpo": "CPO",
    "ppo_lagrangian": "PPO-Lagrangian",
    "budgeted_fitted_q": "Budgeted Fitted-Q",
    "dap_full": "Full DAP",
    "dap_immediate_structured": "Immediate planner",
    "mpc_4": "MPC-4",
    "mpc_8": "MPC-8",
    "dap_black_box_transition": "Black-box transition",
    "dap_actor_distilled": "Actor-distilled",
    "dap_no_budget_horizon": "No budget/horizon",
}
FRONTIER_METHODS = ("dap_calibrated", "double_dqn", "cpo", "ppo_lagrangian", "budgeted_fitted_q")
CONTROL_METHODS = ("dap_full", "dap_immediate_structured", "mpc_4", "mpc_8", "dap_black_box_transition", "dap_actor_distilled", "dap_no_budget_horizon")


def _ci(values: pd.Series) -> tuple[float, float, float]:
    values = values.to_numpy(dtype=float)
    rng = np.random.default_rng(20260805)
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _export(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def _frontier(unit: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharex=False)
    for axis, dataset in zip(axes, ("azure2019", "gentd26")):
        frame = unit[(unit.dataset == dataset) & unit.method.isin(FRONTIER_METHODS)]
        for method in FRONTIER_METHODS:
            rows = []
            for budget, group in frame[frame.method == method].groupby("budget", sort=True):
                x, xl, xh = _ci(group.total_cost)
                y, yl, yh = _ci(group[metric])
                rows.append((budget, x, xl, xh, y, yl, yh))
            plot = np.asarray(rows, dtype=float)
            axis.errorbar(
                plot[:, 1], plot[:, 4], xerr=np.vstack((plot[:, 1] - plot[:, 2], plot[:, 3] - plot[:, 1])),
                yerr=np.vstack((plot[:, 4] - plot[:, 5], plot[:, 6] - plot[:, 4])),
                color=COLORS[method], marker="o", markersize=4, linewidth=2.0 if method == "dap_calibrated" else 1.1,
                capsize=2, label=LABELS[method], zorder=3 if method == "dap_calibrated" else 2,
            )
        axis.set_title("Azure2019" if dataset == "azure2019" else "GenTD26")
        axis.set_xlabel("Observed resource cost")
        axis.grid(axis="both", linewidth=0.4, color="#D9D9D9")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.11))
    fig.subplots_adjust(bottom=0.27, wspace=0.30)
    _export(fig, path)


def _paired_effects(comparisons: pd.DataFrame, path: Path) -> None:
    baselines = ("double_dqn", "cpo", "ppo_lagrangian", "budgeted_fitted_q")
    metrics = (("discounted_return", "Return improvement"), ("total_cost", "Cost reduction"))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    labels = ["Azure: " + LABELS[b] for b in baselines] + ["GenTD26: " + LABELS[b] for b in baselines]
    y = np.arange(len(labels))
    for axis, (metric, title) in zip(axes, metrics):
        rows = []
        for dataset in ("azure2019", "gentd26"):
            for baseline in baselines:
                row = comparisons[(comparisons.dataset == dataset) & (comparisons.baseline == baseline) & (comparisons.metric == metric)].iloc[0]
                rows.append(row)
        values = np.asarray([r["mean"] for r in rows], dtype=float)
        low = np.asarray([r["ci_low"] for r in rows], dtype=float)
        high = np.asarray([r["ci_high"] for r in rows], dtype=float)
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.errorbar(values, y, xerr=np.vstack((values - low, high - values)), fmt="o", color="#0072B2", capsize=2)
        axis.set_title(title)
        axis.grid(axis="x", linewidth=0.4, color="#D9D9D9")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.supxlabel("Favourable paired DAP difference (95% bootstrap CI; n=10 seeds)", fontsize=8, y=0.02)
    fig.subplots_adjust(wspace=0.33, left=0.28, bottom=0.18)
    _export(fig, path)


def _controls(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    for axis, dataset in zip(axes, ("azure2019", "gentd26")):
        frame = summary[(summary.dataset == dataset) & summary.method.isin(CONTROL_METHODS)]
        for method in CONTROL_METHODS:
            row = frame[frame.method == method].iloc[0]
            axis.scatter(
                row.total_cost, row.discounted_return, color=COLORS[method],
                s=36 if method == "dap_full" else 24, zorder=3, label=LABELS[method],
            )
        axis.set_title("Azure2019 validation" if dataset == "azure2019" else "GenTD26 validation")
        axis.set_xlabel("Observed resource cost")
        axis.grid(axis="both", linewidth=0.4, color="#D9D9D9")
    axes[0].set_ylabel("Discounted return")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(wspace=0.30, bottom=0.27)
    _export(fig, path)


def _decision_time(control_unit: pd.DataFrame, path: Path) -> None:
    methods = ("dap_full", "dap_immediate_structured", "mpc_4", "mpc_8")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True)
    for axis, dataset in zip(axes, ("azure2019", "gentd26")):
        frame = control_unit[(control_unit.dataset == dataset) & control_unit.method.isin(methods)]
        positions = np.arange(len(methods))
        for i, method in enumerate(methods):
            mean, low, high = _ci(frame[frame.method == method].decision_ms_mean)
            axis.errorbar(i, mean, yerr=[[mean - low], [high - mean]], fmt="o", color=COLORS[method], capsize=3)
        axis.set_xticks(positions, [LABELS[m] for m in methods], rotation=15, ha="right")
        axis.set_yscale("log")
        axis.set_title("Azure2019 validation" if dataset == "azure2019" else "GenTD26 validation")
        axis.grid(axis="y", linewidth=0.4, color="#D9D9D9")
    axes[0].set_ylabel("Mean decision time (ms, log scale)")
    fig.subplots_adjust(wspace=0.18, bottom=0.27, left=0.10)
    _export(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-name", default="figures_v2")
    args = parser.parse_args()
    root = args.project_root.resolve()
    temporal = root / "results/direct_action_planning_paper_closure/paper_closure_temporal_test_v2_locked/analysis_v1"
    controls = root / "results/direct_action_planning_paper_closure/paper_closure_controls_v3_locked/analysis_v1"
    output = root / "research/direct_action_planning_paper_closure" / args.output_name
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    unit = pd.read_csv(temporal / "unit_metrics.csv")
    paired = pd.read_csv(temporal / "paired_comparisons.csv")
    control_summary = pd.read_csv(controls / "method_summary.csv")
    control_unit = pd.read_csv(controls / "unit_metrics.csv")
    _frontier(unit, "discounted_return", "Discounted return", output / "F1_return_cost_frontier")
    _frontier(unit, "slo_violation_rate", "SLO violation rate", output / "F2_slo_cost_frontier")
    _paired_effects(paired, output / "F3_paired_external_effects")
    _controls(control_summary, output / "F4_same_information_controls_development")
    _decision_time(control_unit, output / "F5_planning_time_development")
    captions = """# Figure Captions

## F1: Return--cost frontier

Locked temporal test evaluation. Each point is the mean of ten independently
trained seeds at one budget; horizontal and vertical bars are 95% bootstrap
intervals over seeds. Lines connect budget levels only and do not imply a
continuous response curve.

## F2: SLO--cost frontier

Locked temporal test evaluation with the same unit and intervals as F1. Lower
SLO violation is better.

## F3: Paired external comparisons

Favourable DAP-minus-baseline differences averaged across the five registered
budgets, one observation per independently trained seed (n=10). Intervals are
95% bootstrap intervals. The formal table reports Wilcoxon p-values and
BH-FDR q-values.

## F4: Same-information controls

Development-only validation data, n=5 registered seeds and budgets 48/96/144.
This figure illustrates effect direction and does not assert FDR-significant
control differences.

## F5: Planning time

Development-only validation data, n=5 registered seeds and budgets 48/96/144.
The y axis is logarithmic because MPC planning spans orders of magnitude; each
point has a 95% bootstrap interval.
"""
    (output / "CAPTIONS.md").write_text(captions, encoding="utf-8")
    write_json(output / "manifest.json", {
        "schema": "stage2.dap_paper_closure.figures.v1",
        "status": "completed",
        "source_analysis": {
            "temporal": str(temporal.relative_to(root)),
            "controls": str(controls.relative_to(root)),
            "temporal_manifest_sha256": sha256_file(temporal / "manifest.json"),
            "controls_manifest_sha256": sha256_file(controls / "manifest.json"),
        },
        "artifacts": {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"},
    })


if __name__ == "__main__":
    main()
