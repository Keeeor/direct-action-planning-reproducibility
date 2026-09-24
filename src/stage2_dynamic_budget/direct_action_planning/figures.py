from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "optimal": "Exact DP",
    "b4_budget_state": "B4",
    "dsp_b": "DSP-B",
    "acba_a": "ACBA-A",
    "oracle_branch": "Oracle-Branch",
    "learned_value_branch": "Learned-Value",
    "learned_model_branch": "Learned-Model",
}
COLORS = {
    "optimal": "#000000",
    "b4_budget_state": "#E69F00",
    "dsp_b": "#0072B2",
    "acba_a": "#CC79A7",
    "oracle_branch": "#666666",
    "learned_value_branch": "#009E73",
    "learned_model_branch": "#56B4E9",
}
MARKERS = {
    "optimal": "o",
    "b4_budget_state": "s",
    "dsp_b": "^",
    "acba_a": "D",
    "oracle_branch": "P",
    "learned_value_branch": "v",
    "learned_model_branch": "X",
}


def _save(fig: plt.Figure, output_dir: Path, name: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{name}.png", output_dir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=240, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _cell_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    rng = np.random.default_rng(20260802)
    indices = rng.integers(0, len(values), size=(5000, len(values)))
    means = values[indices].mean(axis=1)
    return mean, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def plot_policy_metrics(episodes: pd.DataFrame, output_dir: Path) -> list[Path]:
    cells = episodes.groupby(["method", "scenario", "budget", "seed"], as_index=False)[
        ["action_consistency_rate", "mean_Q_star_regret"]
    ].mean()
    methods = list(METHOD_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for axis, metric, label in (
        (axes[0], "action_consistency_rate", "Exact optimal-action agreement"),
        (axes[1], "mean_Q_star_regret", "Mean Q* regret"),
    ):
        for index, method in enumerate(methods):
            values = cells.loc[cells.method == method, metric].to_numpy()
            mean, lower, upper = _cell_interval(values)
            axis.errorbar(
                index,
                mean,
                yerr=[[mean - lower], [upper - mean]],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=6,
                capsize=3,
                linestyle="none",
            )
        axis.set_xticks(
            range(len(methods)),
            [METHOD_LABELS[method] for method in methods],
            rotation=32,
            ha="right",
        )
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_ylim(bottom=0.0)
    fig.suptitle("Minimal validation (mean and paired-cell bootstrap 95% CI, n=45)")
    fig.tight_layout()
    return _save(fig, output_dir, "policy_agreement_and_regret")


def plot_value_model_errors(
    value_diagnostics: pd.DataFrame,
    model_diagnostics: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    scenarios = list(dict.fromkeys(value_diagnostics.scenario))
    for index, scenario in enumerate(scenarios):
        value = value_diagnostics[value_diagnostics.scenario == scenario].value_mae
        model = model_diagnostics[
            model_diagnostics.scenario == scenario
        ].next_state_total_variation_mean
        axes[0].scatter(
            np.full(len(value), index), value, color="#009E73", marker="o", alpha=0.8
        )
        axes[1].scatter(
            np.full(len(model), index), model, color="#0072B2", marker="s", alpha=0.8
        )
    for axis in axes:
        axis.set_xticks(range(len(scenarios)), [item.replace("_", " ") for item in scenarios])
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.set_ylim(bottom=0.0)
    axes[0].set_ylabel("Value MAE against evaluation V*")
    axes[1].set_ylabel("Next-state total variation")
    fig.suptitle("Learned value and one-step model diagnostics (five seeds per scenario)")
    fig.tight_layout()
    return _save(fig, output_dir, "value_and_model_error")


def plot_budget_trajectories(steps: pd.DataFrame, output_dir: Path) -> list[Path]:
    frame = steps.copy()
    frame["remaining_budget_ratio"] = frame.remaining_budget / frame.budget
    summary = frame.groupby(["method", "t"], as_index=False).remaining_budget_ratio.mean()
    fig, axis = plt.subplots(figsize=(7.4, 3.9))
    for method in METHOD_LABELS:
        subset = summary[summary.method == method]
        axis.plot(
            subset.t,
            subset.remaining_budget_ratio,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            markevery=4,
            linewidth=1.4,
        )
    axis.set_xlabel("Decision step")
    axis.set_ylabel("Mean remaining-budget ratio")
    axis.set_ylim(0.0, 1.02)
    axis.grid(color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=4, fontsize=7)
    fig.tight_layout()
    return _save(fig, output_dir, "budget_trajectories")


def plot_service_cost(episodes: pd.DataFrame, output_dir: Path) -> list[Path]:
    summary = episodes.groupby("method")[
        ["total_cost", "completion_rate", "slo_violation_rate"]
    ].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
    for method in METHOD_LABELS:
        row = summary.loc[method]
        for axis, y in ((axes[0], row.completion_rate), (axes[1], row.slo_violation_rate)):
            axis.scatter(
                row.total_cost,
                y,
                color=COLORS[method],
                marker=MARKERS[method],
                s=45,
                label=METHOD_LABELS[method],
            )
    axes[0].set_ylabel("Completion rate (higher is better)")
    axes[1].set_ylabel("SLO violation rate (lower is better)")
    for axis in axes:
        axis.set_xlabel("Mean total resource cost")
        axis.grid(color="#DDDDDD", linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return _save(fig, output_dir, "service_cost_pareto")


def generate_all_figures(
    episodes: pd.DataFrame,
    steps: pd.DataFrame,
    value_diagnostics: pd.DataFrame,
    model_diagnostics: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    paths.extend(plot_policy_metrics(episodes, output_dir))
    paths.extend(plot_value_model_errors(value_diagnostics, model_diagnostics, output_dir))
    paths.extend(plot_budget_trajectories(steps, output_dir))
    paths.extend(plot_service_cost(episodes, output_dir))
    return paths
