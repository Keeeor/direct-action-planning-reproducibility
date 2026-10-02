from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "optimal": "Exact DP",
    "b4_budget_state": "B4",
    "cdba": "CDBA",
    "dsp_a": "DSP-A",
    "dsp_b": "DSP-B",
    "acba_a": "ACBA-A",
    "acba_b": "ACBA-B",
}
COLORS = {
    "optimal": "#000000",
    "b4_budget_state": "#E69F00",
    "cdba": "#D55E00",
    "dsp_a": "#0072B2",
    "dsp_b": "#56B4E9",
    "acba_a": "#009E73",
    "acba_b": "#CC79A7",
}


def _save(fig: plt.Figure, output_dir: Path, name: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{name}.png", output_dir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=240, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_optimal_action_heatmaps(truth: pd.DataFrame, output_dir: Path) -> list[Path]:
    scenarios = list(dict.fromkeys(truth.scenario))
    horizon = int(truth.remaining_horizon.max())
    target_horizon = max(horizon // 2, 1)
    load = int(truth.load.max())
    cmap = ListedColormap(["#F0F0F0", "#56B4E9", "#E69F00", "#009E73"])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    image = None
    for axis, scenario in zip(axes, scenarios):
        subset = truth[
            (truth.scenario == scenario)
            & (truth.remaining_horizon == target_horizon)
            & (truth.load == load)
            & (truth.action == 0)
        ]
        matrix = subset.pivot(index="queue", columns="remaining_budget", values="optimal_action")
        image = axis.imshow(
            matrix.to_numpy(), origin="lower", aspect="auto", cmap=cmap, norm=norm
        )
        axis.set_title(scenario.replace("_", " "))
        axis.set_xlabel("Remaining budget")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns)
        axis.set_yticks(range(len(matrix.index)), matrix.index)
    axes[0].set_ylabel("Queue")
    assert image is not None
    colorbar_axis = fig.add_axes([0.91, 0.18, 0.018, 0.64])
    colorbar = fig.colorbar(image, cax=colorbar_axis, ticks=range(4))
    colorbar.set_label("Optimal action")
    fig.suptitle(f"Exact optimal action, load={load}, remaining horizon={target_horizon}")
    fig.subplots_adjust(top=0.82, right=0.88, wspace=0.18)
    return _save(fig, output_dir, "optimal_action_heatmaps")


def plot_action_values(truth: pd.DataFrame, output_dir: Path) -> list[Path]:
    feasible = truth[truth.feasible]
    summary = feasible.groupby(["scenario", "action", "remaining_budget"], as_index=False)[
        ["Q_star", "A_star"]
    ].mean()
    scenarios = list(dict.fromkeys(summary.scenario))
    fig, axes = plt.subplots(2, len(scenarios), figsize=(4.4 * len(scenarios), 6.0), sharex=True)
    axes = np.asarray(axes).reshape(2, len(scenarios))
    action_colors = ["#000000", "#0072B2", "#E69F00", "#009E73"]
    for column, scenario in enumerate(scenarios):
        subset = summary[summary.scenario == scenario]
        for action, action_frame in subset.groupby("action"):
            for row, metric in enumerate(("Q_star", "A_star")):
                axes[row, column].plot(
                    action_frame.remaining_budget,
                    action_frame[metric],
                    label=f"action {action}",
                    color=action_colors[int(action)],
                    linewidth=1.6,
                )
        axes[0, column].set_title(scenario.replace("_", " "))
        axes[1, column].set_xlabel("Remaining budget")
        axes[0, column].axhline(0, color="#999999", linewidth=0.7)
        axes[1, column].axhline(0, color="#999999", linewidth=0.7)
    axes[0, 0].set_ylabel("Mean Q*")
    axes[1, 0].set_ylabel("Mean A* relative to a0")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Action-conditioned exact values")
    fig.tight_layout()
    return _save(fig, output_dir, "action_q_and_advantage")


def plot_confusions(state_rows: pd.DataFrame, output_dir: Path) -> list[Path]:
    methods = [method for method in METHOD_LABELS if method != "optimal"]
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.8), sharex=True, sharey=True)
    matrices: dict[str, np.ndarray] = {}
    for method in methods:
        subset = state_rows[state_rows.method == method]
        matrix = pd.crosstab(subset.optimal_action, subset.action).reindex(
            index=range(4), columns=range(4), fill_value=0
        )
        normalized = matrix.to_numpy(dtype=float)
        normalized = normalized / np.maximum(normalized.sum(axis=1, keepdims=True), 1)
        matrices[method] = normalized
    image = None
    for axis, method in zip(axes.flat, methods):
        image = axis.imshow(matrices[method], vmin=0, vmax=1, cmap="Blues", origin="upper")
        axis.set_title(METHOD_LABELS[method])
        axis.set_xticks(range(4))
        axis.set_yticks(range(4))
        for row, column in np.ndindex(4, 4):
            value = matrices[method][row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.55 else "black",
            )
    for axis in axes[-1]:
        axis.set_xlabel("Selected action")
    for axis in axes[:, 0]:
        axis.set_ylabel("Optimal action")
    assert image is not None
    colorbar_axis = fig.add_axes([0.91, 0.16, 0.018, 0.68])
    fig.colorbar(image, cax=colorbar_axis, label="Row-normalized frequency")
    fig.suptitle("Action confusion against exact DP")
    fig.subplots_adjust(top=0.88, right=0.88, wspace=0.22, hspace=0.28)
    return _save(fig, output_dir, "method_action_confusions")


def plot_policy_metrics(episodes: pd.DataFrame, output_dir: Path) -> list[Path]:
    methods = list(METHOD_LABELS)
    summary = episodes.groupby("method")[["action_consistency_rate", "mean_Q_star_regret"]].agg(
        ["mean", "sem"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    x = np.arange(len(methods))
    for axis, metric, label in (
        (axes[0], "action_consistency_rate", "Action agreement"),
        (axes[1], "mean_Q_star_regret", "Mean exact Q* regret"),
    ):
        means = [summary.loc[method, (metric, "mean")] for method in methods]
        errors = [summary.loc[method, (metric, "sem")] for method in methods]
        axis.bar(x, means, yerr=errors, color=[COLORS[method] for method in methods], capsize=2)
        axis.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=35, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylim(0, 1.02)
    fig.suptitle("Paired minimal-validation policy metrics")
    fig.tight_layout()
    return _save(fig, output_dir, "policy_agreement_and_regret")


def plot_pareto(episodes: pd.DataFrame, output_dir: Path) -> list[Path]:
    summary = episodes.groupby("method")[["total_cost", "completion_rate", "slo_violation_rate"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for method in METHOD_LABELS:
        row = summary.loc[method]
        axes[0].scatter(row.total_cost, row.completion_rate, color=COLORS[method], s=42)
        axes[1].scatter(row.total_cost, row.slo_violation_rate, color=COLORS[method], s=42)
    axes[0].set_xlabel("Mean total resource cost")
    axes[0].set_ylabel("Completion rate (higher is better)")
    axes[1].set_xlabel("Mean total resource cost")
    axes[1].set_ylabel("SLO violation rate (lower is better)")
    for axis in axes:
        axis.grid(color="#DDDDDD", linewidth=0.6)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=COLORS[method],
            label=METHOD_LABELS[method],
            markersize=6,
        )
        for method in METHOD_LABELS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=7,
        frameon=False,
        fontsize=7,
    )
    fig.suptitle("Service-cost relation", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    return _save(fig, output_dir, "service_cost_pareto")


def plot_budget_trajectories(steps: pd.DataFrame, output_dir: Path) -> list[Path]:
    frame = steps.copy()
    frame["remaining_budget_ratio"] = frame.remaining_budget / frame.budget
    summary = frame.groupby(["method", "t"], as_index=False).remaining_budget_ratio.mean()
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    for method in METHOD_LABELS:
        subset = summary[summary.method == method]
        axis.plot(
            subset.t,
            subset.remaining_budget_ratio,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            linewidth=1.5,
        )
    axis.set_xlabel("Decision step")
    axis.set_ylabel("Mean remaining budget ratio")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=4, fontsize=8)
    axis.set_title("Budget-use trajectories")
    fig.tight_layout()
    return _save(fig, output_dir, "budget_trajectories")


def generate_all_figures(
    truth: pd.DataFrame,
    state_rows: pd.DataFrame,
    episodes: pd.DataFrame,
    steps: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    paths.extend(plot_optimal_action_heatmaps(truth, output_dir))
    paths.extend(plot_action_values(truth, output_dir))
    paths.extend(plot_confusions(state_rows, output_dir))
    paths.extend(plot_policy_metrics(episodes, output_dir))
    paths.extend(plot_pareto(episodes, output_dir))
    paths.extend(plot_budget_trajectories(steps, output_dir))
    return paths
